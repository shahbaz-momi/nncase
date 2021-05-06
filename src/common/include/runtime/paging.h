#pragma once
#include "model.h"

/**
 * kmodel flag which enables paging for this model.
 */
#define KM_ENABLE_PAGING 0x02

/**
 * Maximum number of possible pages.
 */
#define KM_MAX_PAGES 8

/**
 * The target size for each page, in number of bytes.
 */
#define TARGET_PAGE_SIZE 2300000

enum memory_page_type : uint32_t {
    persistent = 0,
    swap = 1,
};

/**
 * Defines a page of a kmodel body. Note that constants and main memory is not paged; only the model
 * body itself.
 */
typedef struct {
    /**
     * Index of this page.
     */
    uint32_t index;
    /**
     * Type of this page. Persistent pages are always kept in memory, swap pages are assumed to be independent
     * of each other and
     */
    memory_page_type type;
    /**
     * The beginning of the range of nodes this page covers.
     */
    uint32_t begin;
    /**
     * The end of the range of the nodes this pages covers, inclusive.
     */
    uint32_t end;
    /**
     * The offset from the beginning of the body of the kmodel to the contents of this page. Used for loading
     * from flash.
     */
    uint64_t offset_bytes;
    /**
     * The size (in bytes) of the contents of the node bodies in this page. Used for loading from flash.
     */
    uint64_t size_bytes;
} memory_page;

typedef struct {
    /**
     * The total number of pages.
     */
    uint32_t num_pages;
    /**
     * The maximum number of pages stored in this model (used for calculating loading offsets).
     */
    uint32_t max_pages;
    /**
     * The size of the required to execute the model. Includes all persistent pages plus the largest swap page.
     */
    uint64_t body_buffer_size;
} memory_page_table;