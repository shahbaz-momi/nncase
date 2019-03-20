﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;

#if NET471
using System.Collections.Async;
#endif

namespace NnCase.Converter.Data
{
    public enum PostprocessMethods
    {
        None,
        Normalize0To1,
        NormalizeMinus1To1,
        Whitening
    }

    public enum PreprocessMethods
    {
        None,
        Darknet
    }

    public abstract class Dataset
    {
        private readonly IReadOnlyList<string> _fileNames;
        private readonly int[] _dimensions;
        private readonly int _batchSize;

        public PostprocessMethods PostprocessMethod { get; }

        public ReadOnlySpan<int> Dimensions => _dimensions;

        public Dataset(string path, IReadOnlyCollection<string> allowdExtensions, ReadOnlySpan<int> dimensions, int batchSize, PostprocessMethods postprocessMethod)
        {
            if (batchSize < 1)
                throw new ArgumentOutOfRangeException(nameof(batchSize));

            if (Directory.Exists(path))
            {
                _fileNames = (from f in Directory.EnumerateFiles(path, "*.*", SearchOption.AllDirectories)
                              where allowdExtensions.Contains(Path.GetExtension(f).ToLowerInvariant())
                              select f).ToList();
            }
            else if (allowdExtensions.Contains(Path.GetExtension(path)))
            {
                _fileNames = new[] { path };
            }

            if (_fileNames == null || _fileNames.Count == 0)
            {
                throw new ArgumentException("Invalid dataset.");
            }

            _dimensions = dimensions.ToArray();
            _batchSize = batchSize;
            PostprocessMethod = postprocessMethod;
        }


        public
#if NET471
#else
            async
#endif
            IAsyncEnumerable<Tensor<float>> GetBatchesAsync()
        {
            var dimensions = new[] { _batchSize }.Concat(_dimensions).ToArray();
            var oneSize = Dimensions.GetSize();

            async Task<byte[]> ReadAllBytesAsync(string path)
            {
                using (var fs = File.OpenRead(path))
                {
                    var buffer = new byte[(int)fs.Length];
                    await fs.ReadAsync(buffer, 0, buffer.Length);
                    return buffer;
                }
            }

#if NET471
            return new AsyncEnumerable<Tensor<float>>(async yield =>
            {
#endif
            for (int i = 0; i < _fileNames.Count / _batchSize; i++)
            {
                var tensor = new DenseTensor<float>(dimensions);
                var sources = await Task.WhenAll(from j in Enumerable.Range(i * _batchSize, _batchSize)
                                                 select ReadAllBytesAsync(_fileNames[j]));
                Parallel.For(0, _batchSize, j =>
                {
                    var buffer = tensor.Buffer.Slice(j * oneSize);
                    Process(sources[j], buffer.Span);
                    Postprocess(buffer.Span, PostprocessMethod);
                });

#if NET471
                    await yield.ReturnAsync(tensor);
#else
                yield return tensor;
#endif
            }
#if NET471
            });
#endif
        }

        public
#if NET471
#else
            async
#endif
            IAsyncEnumerable<Tensor<byte>> GetFixedBatchesAsync()
        {
            var dimensions = new[] { _batchSize }.Concat(_dimensions).ToArray();
            var oneSize = Dimensions.GetSize();

            async Task<byte[]> ReadAllBytesAsync(string path)
            {
                using (var fs = File.OpenRead(path))
                {
                    var buffer = new byte[(int)fs.Length];
                    await fs.ReadAsync(buffer, 0, buffer.Length);
                    return buffer;
                }
            }

#if NET471
            return new AsyncEnumerable<Tensor<byte>>(async yield =>
            {
#endif
            for (int i = 0; i < _fileNames.Count / _batchSize; i++)
            {
                var tensor = new DenseTensor<byte>(dimensions);
                var sources = await Task.WhenAll(from j in Enumerable.Range(i * _batchSize, _batchSize)
                                                 select ReadAllBytesAsync(_fileNames[j]));
                Parallel.For(0, _batchSize, j =>
                {
                    var buffer = tensor.Buffer.Slice(j * oneSize);
                    Process(sources[j], buffer.Span);
                });

#if NET471
                    await yield.ReturnAsync(tensor);
#else
                yield return tensor;
#endif
            }
#if NET471
            });
#endif
        }

        private void Postprocess(Span<float> data, PostprocessMethods postprocessMethod)
        {
            if (postprocessMethod == PostprocessMethods.Normalize0To1)
            {
            }
            else if (postprocessMethod == PostprocessMethods.NormalizeMinus1To1)
            {
                for (int i = 0; i < data.Length; i++)
                    data[i] = data[i] * 2f - 1f;
            }
            else if (postprocessMethod != PostprocessMethods.None)
                throw new NotSupportedException(nameof(postprocessMethod));
        }

        protected abstract void Process(byte[] source, Span<float> dest);
        protected abstract void Process(byte[] source, Span<byte> dest);
    }

    public class ImageDataset : Dataset
    {
        private static readonly string[] _allowdExtensions = new[]
        {
            ".bmp", ".jpg", ".png"
        };

        private readonly PreprocessMethods _preprocessMethods;

        public ImageDataset(string path, ReadOnlySpan<int> dimensions, int batchSize, PreprocessMethods preprocessMethods, PostprocessMethods postprocessMethod)
            : base(path, _allowdExtensions, dimensions, batchSize, postprocessMethod)
        {
            _preprocessMethods = preprocessMethods;
        }

        protected Image<Rgb24> Process(byte[] source)
        {
            using (var image = Image.Load<Rgb24>(source))
            {
                Image<Rgb24> destImage;
                if (_preprocessMethods == PreprocessMethods.Darknet)
                {
                    image.Mutate(x =>
                        x.Resize(new ResizeOptions
                        {
                            Size = new Size(Dimensions[2], Dimensions[1]),
                            Sampler = KnownResamplers.Bicubic,
                            Position = AnchorPositionMode.Center,
                            Mode = ResizeMode.Max
                        }));
                    destImage = new Image<Rgb24>(image.GetConfiguration(), Dimensions[2], Dimensions[1], new Rgb24(127, 127, 127));
                    var leftTop = new Point((destImage.Width - image.Width) / 2, (destImage.Height - image.Height) / 2);
                    destImage.Mutate(x =>
                        x.DrawImage(image, leftTop, GraphicsOptions.Default));
                }
                else
                {
                    image.Mutate(x =>
                        x.Resize(Dimensions[2], Dimensions[1]));
                    destImage = image.Clone();
                }

                return destImage;
            }
        }

        protected override void Process(byte[] source, Span<float> dest)
        {
            using (var destImage = Process(source))
            {
                if (Dimensions[0] == 3)
                {
                    var pixels = destImage.GetPixelSpan();
                    var channelSize = Dimensions[1] * Dimensions[2];

                    var rChannel = dest.Slice(0, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        rChannel[i] = pixels[i].R / 255.0f;

                    var gChannel = dest.Slice(channelSize, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        gChannel[i] = pixels[i].G / 255.0f;

                    var bChannel = dest.Slice(channelSize * 2, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        bChannel[i] = pixels[i].B / 255.0f;
                }
                else if (Dimensions[0] == 1)
                {
                    destImage.Mutate(x =>
                        x.Grayscale());

                    var pixels = destImage.GetPixelSpan();
                    var channelSize = Dimensions[1] * Dimensions[2];

                    var rChannel = dest.Slice(0, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        rChannel[i] = pixels[i].R / 255.0f;
                }
                else
                {
                    throw new NotSupportedException($"Channels number {Dimensions[0]} is not supported by dataset provider.");
                }
            }
        }

        protected override void Process(byte[] source, Span<byte> dest)
        {
            using (var destImage = Process(source))
            {
                if (Dimensions[0] == 3)
                {
                    var pixels = destImage.GetPixelSpan();
                    var channelSize = Dimensions[1] * Dimensions[2];

                    var rChannel = dest.Slice(0, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        rChannel[i] = pixels[i].R;

                    var gChannel = dest.Slice(channelSize, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        gChannel[i] = pixels[i].G;

                    var bChannel = dest.Slice(channelSize * 2, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        bChannel[i] = pixels[i].B;
                }
                else if (Dimensions[0] == 1)
                {
                    destImage.Mutate(x =>
                        x.Grayscale());

                    var pixels = destImage.GetPixelSpan();
                    var channelSize = Dimensions[1] * Dimensions[2];

                    var rChannel = dest.Slice(0, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        rChannel[i] = pixels[i].R;
                }
                else
                {
                    throw new NotSupportedException($"Channels number {Dimensions[0]} is not supported by dataset provider.");
                }
            }
        }
    }
}
