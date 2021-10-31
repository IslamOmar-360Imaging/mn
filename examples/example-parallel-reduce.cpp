#include <mn/IO.h>
#include <mn/Fabric.h>
#include <mn/Defer.h>
#include <mn/Buf.h>

#include <fmt/chrono.h>

#include <chrono>

static constexpr const size_t max_array_size = 10e5;
static constexpr const size_t max_pixel_value = 10e5;


inline static mn::Buf<int>
histo1(const mn::Buf<uint64_t>& pixels)
{
	auto histogram = mn::buf_with_allocator<int>(mn::memory::tmp());
	mn::buf_resize_fill(histogram,max_array_size , 0);

	for (auto p: pixels)
		++histogram[p];

	return histogram;
}

inline static mn::Buf<int>
histo2(const mn::Buf<uint64_t>& pixels, mn::Fabric f)
{
	auto histogram = mn::buf_with_allocator<int>(mn::memory::tmp());
	mn::buf_resize_fill(histogram,max_array_size , 0);

	mn::compute(f, {pixels.count, 1, 1}, {262144, 1, 1}, [&](mn::Compute_Args args){
		for (size_t i = 0; i < args.tile_size.x; ++i)
			++histogram[pixels[args.global_invocation_id.x + i]];
	});

	return histogram;
}

inline static mn::Buf<int>
histo3(const mn::Buf<uint64_t>& pixels, mn::Fabric f)
{
	auto histogram = mn::buf_with_allocator<int>(mn::memory::tmp());
	mn::buf_resize_fill(histogram, max_array_size, 0);

	auto mtx = mn::mutex_new();
	mn_defer(mn::mutex_free(mtx));

	mn::compute(f, {pixels.count, 1, 1}, {262144, 1, 1}, [&](mn::Compute_Args args){
		mn::mutex_lock(mtx);
		mn_defer(mn::mutex_unlock(mtx));

		for (size_t i = 0; i < args.tile_size.x; ++i)
			++histogram[pixels[args.global_invocation_id.x + i]];
	});

	return histogram;
}

inline static mn::Buf<int>
histo4(const mn::Buf<uint64_t>& pixels, mn::Fabric f)
{
	auto histogram = mn::buf_with_allocator<int>(mn::memory::tmp());
	mn::buf_resize_fill(histogram, max_array_size, 0);

	auto mtxs = mn::buf_with_count<mn::Mutex>(histogram.count);
	for (size_t i = 0; i < mtxs.count; ++i)
		mtxs[i] = mn::mutex_new();
	mn_defer(destruct(mtxs));

	mn::compute(f, {pixels.count, 1, 1}, {262144, 1, 1}, [&](mn::Compute_Args args){
		for (size_t i = 0; i < args.tile_size.x; ++i)
		{
			auto v = pixels[args.global_invocation_id.x + i];
			mn::mutex_lock(mtxs[v]);
			++histogram[v];
			mn::mutex_unlock(mtxs[v]);
		}
	});

	return histogram;
}

inline static mn::Buf<std::atomic<int>>
histo5(const mn::Buf<uint64_t>& pixels, mn::Fabric f)
{
	auto histogram = mn::buf_with_allocator<std::atomic<int>>(mn::memory::tmp());
	mn::buf_resize_fill(histogram, max_array_size, 0);

	mn::compute(f, {pixels.count, 1, 1}, {262144, 1, 1}, [&](mn::Compute_Args args){
		for (size_t i = 0; i < args.tile_size.x; ++i)
		{
			++histogram[pixels[args.global_invocation_id.x + i]];
		}
	});

	return histogram;
}

inline static mn::Buf<int>
histo6(const mn::Buf<uint64_t>& pixels, mn::Fabric f)
{
	mn::Buf<mn::Buf<int>> histograms = mn::buf_new<mn::Buf<int>>();
	mn::buf_resize_fill(histograms, mn::fabric_workers_count(f), mn::buf_with_allocator<int>(mn::memory::tmp()));
	for (size_t i = 0; i < histograms.count; ++i)
		mn::buf_resize_fill(histograms[i], (max_array_size) * 2, 0);
	mn_defer(mn::buf_free(histograms));

	mn::compute(f, {pixels.count, 1, 1}, {262144, 1, 1}, [&](mn::Compute_Args args){
		auto histogram = histograms[mn::local_worker_index()];

		auto begin = pixels.ptr + args.global_invocation_id.x;
		auto end = begin + args.tile_size.x;
		for (auto it = begin; it != end; ++it)
			++histogram[*it];
	});

	auto histogram = mn::buf_with_allocator<int>(mn::memory::tmp());
	mn::buf_resize_fill(histogram, max_array_size, 0);

	for (auto h: histograms)
		for (size_t i = 0; i < histogram.count; ++i)
			histogram[i] += h[i];

	return histogram;
}

int main()
{
	auto f = mn::fabric_new({});
	mn_defer(mn::fabric_free(f));

	auto pixels = mn::buf_with_allocator<uint64_t>(mn::memory::tmp());
	mn::buf_resize(pixels, 512ULL * 512ULL * 512ULL);

	for (auto& p: pixels)
		p = rand() % max_pixel_value;

	size_t times = 3;
	auto res1 = histo1(pixels);
	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < times; ++i)
		auto res1 = histo1(pixels);
	auto end = std::chrono::high_resolution_clock::now();
	mn::print("histo1: {}\n", std::chrono::duration<double, std::milli>(end - start) / times);

	auto res2 = histo2(pixels, f);
	start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < times; ++i)
		auto res2 = histo2(pixels, f);
	end = std::chrono::high_resolution_clock::now();
	mn::print("histo2: {}\n", std::chrono::duration<double, std::milli>(end - start) / times);

	auto res3 = histo3(pixels, f);
	start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < times; ++i)
		auto res3 = histo3(pixels, f);
	end = std::chrono::high_resolution_clock::now();
	mn::print("histo3: {}\n", std::chrono::duration<double, std::milli>(end - start) / times);

	auto res4 = histo4(pixels, f);
	start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < times; ++i)
		auto res4 = histo4(pixels, f);
	end = std::chrono::high_resolution_clock::now();
	mn::print("histo4: {}\n", std::chrono::duration<double, std::milli>(end - start) / times);

	auto res5 = histo5(pixels, f);
	start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < times; ++i)
		auto res5 = histo5(pixels, f);
	end = std::chrono::high_resolution_clock::now();
	mn::print("histo5: {}\n", std::chrono::duration<double, std::milli>(end - start) / times);

	auto res6 = histo6(pixels, f);
	start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < times; ++i)
		auto res6 = histo6(pixels, f);
	end = std::chrono::high_resolution_clock::now();
	mn::print("histo6: {}\n", std::chrono::duration<double, std::milli>(end - start) / times);

	return 0;
}
