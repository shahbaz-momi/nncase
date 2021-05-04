/* Copyright 2019-2020 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <codegen/codegen.h>
#include <llir/op_utils.h>
#include <llir/ops/constant.h>
#include <runtime/model.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <runtime/paging.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::llir;
using namespace nncase::scheduler;
using namespace nncase::runtime;

namespace
{
std::unordered_map<node_opcode, emitter_t> g_emitters;
std::unordered_set<node_opcode> g_disabled_emitters;

std::unique_ptr<node_body> call_emitter(node &node, codegen_context &context)
{
    auto opcode = node.runtime_opcode();
    auto it = g_emitters.find(opcode);
    if (it == g_emitters.end())
    {
        if (g_disabled_emitters.find(opcode) == g_disabled_emitters.end())
            throw std::runtime_error(std::string("Emitter for ") + node_opcode_names(opcode).data() + " is not found");
    }
    else
    {
        return it->second(node, context);
    }

    return nullptr;
}
}

void nncase::codegen::register_emitter(node_opcode opcode, emitter_t emitter)
{
    g_emitters.emplace(opcode, std::move(emitter));
}

void nncase::codegen::disable_emitter(llir::node_opcode opcode)
{
    g_disabled_emitters.emplace(opcode);
}

codegen_context::codegen_context(std::ostream &output, const std::unordered_map<memory_type_t, memory_allocator *> &allocators, const std::unordered_map<llir::output_connector *, memory_allocation> &allocations)
    : writer_(output), allocators_(allocators), allocations_(allocations)
{
}

memory_range codegen_context::get_allocation(output_connector &conn) const
{
    auto &alloc = allocations_.at(&conn);
    return { alloc.type, conn.type(), (uint32_t)alloc.start, (uint32_t)alloc.size };
}

std::ostream& operator<<(std::ostream& ostream, const memory_page &page) {
    return ostream << "page{index=" << page.index << ",type=" << page.type << ",begin=" << page.begin << ",.end=" << page.end
        << ",size_bytes=" << page.size_bytes;
}

void write_pages(runtime::binary_writer writer, const std::vector<node_header> &headers, const std::vector<llir::node *> &nodes) {
    std::vector<memory_page> pages;

    // We can always hold the first page in memory
    memory_page current = {
            .index = 0,
            .type = persistent,
            .begin = 0, // default to including first node
            .end = 0,
            .offset_bytes = 0,
            .size_bytes = headers[0].body_size
    };

    // Try and include nodes sequentially to hit our target size; try not to go over our target
    for(uint32_t node = 1; node < headers.size(); node ++) {
        if(headers[node].body_size + current.size_bytes > TARGET_PAGE_SIZE) {
            // Commit current page
            pages.emplace_back(current);
            current = {
                    .index = current.index + 1,
                    .type = swap,
                    .begin = node,
                    .end = node,
                    .offset_bytes = current.offset_bytes + current.size_bytes,
                    .size_bytes = headers[node].body_size
            };
        } else {
            // Shift end to current
            current.end = node;
            current.size_bytes += headers[node].body_size;
        }
    }

    // commit last remaining page
    pages.emplace_back(current);

    assert(pages.size() <= KM_MAX_PAGES && "Max number of pages exceeded");

    uint64_t working_size = 0;
    uint64_t largest_swap = 0;
    for(auto &page : pages) {
        if(page.type == persistent) {
            working_size += page.size_bytes;
        } else { // swap
            largest_swap = std::max(largest_swap, page.size_bytes);
        }
    }
    working_size += largest_swap;

    memory_page_table table = {
            .num_pages = (uint32_t) pages.size(),
            .max_pages = KM_MAX_PAGES,
            .body_buffer_size = working_size
    };

    // Write the table first, then pages
    writer.write(table);
    std::cout << "Using pages:" << std::endl;
    for(auto &page : pages) {
        writer.write(page);
        std::cout << "    " << page << std::endl;
    }
    std::cout << "Resident model size: " << std::to_string(table.body_buffer_size) << std::endl;
}

void nncase::codegen::gencode(codegen_context &context, xtl::span<llir::node *> compute_sequence)
{
    std::vector<llir::node *> runtime_nodes;
    std::vector<memory_range> inputs;
    std::vector<runtime_shape_t> input_shapes;
    std::vector<memory_range> outputs;
    std::vector<llir::node *> constants;

    for (auto &&node : compute_sequence)
    {
        if (g_disabled_emitters.find(node->runtime_opcode()) == g_disabled_emitters.end())
            runtime_nodes.emplace_back(node);

        switch (node->runtime_opcode())
        {
        case op_input_node:
            inputs.emplace_back(context.get_allocation(node->output_at(0)));
            input_shapes.emplace_back(llir::to(node->output_at(0).shape()));
            break;
        case op_output_node:
            outputs.emplace_back(context.get_allocation(*node->input_at(0).connection()));
            break;
        case op_constant:
            constants.emplace_back(node);
            break;
        }
    }

    bool enable_paging = true;

    auto &writer = context.writer();
    // model header
    model_header model_header;
    model_header.identifier = MODEL_IDENTIFIER;
    model_header.version = MODEL_VERSION;
    model_header.flags = 0;
    model_header.target = MODEL_TARGET_K210;
    model_header.constants = context.constant_usage();
    model_header.main_mem = context.memory_usage();
    model_header.nodes = runtime_nodes.size();
    model_header.inputs = inputs.size();
    model_header.outputs = outputs.size();

    if(enable_paging) {
        model_header.flags |= KM_NODE_PAGING;
    }

    writer.write(model_header);

    // inputs
    writer.write_array<memory_range>(inputs);
    // input shapes
    writer.write_array<runtime_shape_t>(input_shapes);
    // outputs
    writer.write_array<memory_range>(outputs);

    // constants
    auto const_mem = std::make_unique<uint8_t[]>(context.constant_usage());
    for (auto &node : constants)
    {
        auto &con = static_cast<constant &>(*node);
        auto alloc = context.get_allocation(con.output());
        auto start = const_mem.get() + alloc.start;
        std::copy(con.data().begin(), con.data().end(), start);
    }

    writer.write_array(xtl::span<const uint8_t> { const_mem.get(), context.constant_usage() });

    // Keep node headers
    std::vector<node_header> node_headers;
    auto node_headers_pos = writer.position();
    std::streamoff node_header_bytes = sizeof(node_header) * runtime_nodes.size();

    std::streamoff page_bytes = sizeof(memory_page) * KM_MAX_PAGES * ((model_header.flags & KM_MAX_PAGES)? 1 : 0);

    writer.position(node_headers_pos + node_header_bytes + page_bytes);

    // write body
    for (auto &&node : runtime_nodes)
    {
        auto body = call_emitter(*node, context);
        if (body)
        {
            auto body_start = writer.position();
            body->serialize(writer);
            writer.align_position(8);
            auto body_size = writer.position() - body_start;
            node_headers.emplace_back(node_header { body->opcode(), (uint32_t)body_size });
        }
    }

    // Write node headers
    auto end_pos = writer.position();
    writer.position(node_headers_pos);
    writer.write_array<node_header>(node_headers);

    // Write our pages structure
    if(model_header.flags & KM_NODE_PAGING) {
        write_pages(writer, node_headers, runtime_nodes);
    }

    writer.position(end_pos);

    std::cout << "Working memory usage: " << context.memory_usage() << " B" << std::endl;
}
