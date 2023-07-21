#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "input_generation.h"
#include "definitions.h"
#include "result_collector.h"
#include "utilities.h"

#include "cuda_profiler_api.h"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <iterator>
#include <thread>


template <typename key_type, typename value_type>
size_t estimate_pq_memory_consumption(size_t build_size, size_t probe_size) {
    size_t input_output_memory_consumption = (build_size + probe_size) * (sizeof(key_type) + sizeof(value_type));
    size_t estimated_index_memory_consumption = build_size * std::max(9 * sizeof(float), 2 * (sizeof(key_type) + sizeof(rti_idx)));
    size_t estimated_auxiliary_memory_consumption = std::max(estimated_index_memory_consumption, probe_size * (sizeof(key_type) + sizeof(rti_idx)) * 13 / 10);
    return input_output_memory_consumption + estimated_index_memory_consumption + estimated_auxiliary_memory_consumption;
}


template <typename key_type, typename value_type>
size_t estimate_rq_memory_consumption(size_t build_size, size_t probe_size) {
    size_t base = estimate_pq_memory_consumption<key_type, value_type>(build_size, probe_size);
    size_t additional_input_memory_consumption = probe_size * sizeof(key_type);
    size_t additional_auxiliary_memory_consumption = additional_input_memory_consumption * 13 / 10;
    return base + additional_input_memory_consumption + additional_auxiliary_memory_consumption;
}


struct test_configuration {
    std::string description;
    std::vector<size_t> log_build_size_options;
    std::vector<size_t> log_probe_size_options;
    std::vector<double> hit_rate_options;
    std::vector<bool> misses_are_outliers_options;
    std::vector<size_t> log_num_batches_options;
    std::vector<double> build_key_uniformity_options;
    std::vector<double> probe_zipf_coefficient_options;
    std::vector<size_t> log_key_multiplicity_options;
    std::vector<bool> sort_insert_options;
    std::vector<bool> sort_probe_options;
};


// for nvtx
struct nvtx_benchmark_domain{ static constexpr char const* name{"benchmark"}; };


template <typename index_type, typename value_type>
void benchmark_point_query(
        rc::result_collector& rc,
        size_t runs,
        bool check_all_results,
        bool run_advanced_tests
) {

    nvtx3::scoped_range_in<nvtx_benchmark_domain> index_experiment{index_type::short_description};

    using key_type = typename index_type::key_type;
    constexpr size_t max_build_size = size_t(1) << 26u;

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

#if 0
#warning "a test setup is configured, pre-configured benchmarks will be skipped"
    // NVTX FILTER TO INCLUDE: benchmark@run-2/*/batch-0/lookup-batch/
    // NVTX FILTER TO EXCLUDE: rtx@upload-params/
    std::vector<test_configuration> test_configuration_options {
            {
                    "test",
                    {26}, // log_build_size_options
                    //{15, 16, 25, 26}, // log_build_size_options [N-COMPUTE 2023/07/08]
                    {27}, // log_probe_size_options
                    //{13, 15, 17, 19, 21, 23, 25, 27}, // log_probe_size_options [N-COMPUTE 2023/07/03]
                    {1.0}, // hit_rate_options
                    //{1.0, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01}, // hit_rate_options [N-COMPUTE 2023/07/03]
                    {false}, // misses_are_outliers_options
                    {0}, // log_num_batches_options
                    //{0, 12, 16}, // log_num_batches_options [N-SYSTEMS 2023/07/04]
                    {1.0}, // build_key_uniformity_options
                    {0.0}, // probe_zipf_coefficient_options
                    //{0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, // probe_zipf_coefficient_options [N-COMPUTE 2023/07/03]
                    {0}, // log_key_multiplicity_options
                    {false}, // sort_insert_options
                    {false} // sort_probe_options
                    //{true} // sort_probe_options [N-SYSTEMS 2023/07/06]
                    //{false, true} // sort_probe_options [N-COMPUTE 2023/07/03]
            }
    };
#else
    std::vector<test_configuration> test_configuration_options {
        {
                    "build_size",
                    {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}, // log_build_size_options
                    {27}, // log_probe_size_options
                    {1.0}, // hit_rate_options
                    {false}, // misses_are_outliers_options
                    {0}, // log_num_batches_options
                    {1.0}, // build_key_uniformity_options
                    {0.0}, // probe_zipf_coefficient_options
                    {0}, // log_key_multiplicity_options
                    {false, true}, // sort_insert_options
                    {false, true} // sort_probe_options
        }
    };
    std::vector<test_configuration> advanced_test_configuration_options {
            {
                    "probe_size",
                    {25, 26}, // log_build_size_options
                    {13, 15, 17, 19, 21, 23, 25, 27}, // log_probe_size_options
                    {1.0}, // hit_rate_options
                    {false}, // misses_are_outliers_options
                    {0}, // log_num_batches_options
                    {1.0}, // build_key_uniformity_options
                    {0.0}, // probe_zipf_coefficient_options
                    {0}, // log_key_multiplicity_options
                    {false}, // sort_insert_options
                    {false, true} // sort_probe_options
            },
            {
                    "hit_rate",
                    {26}, // log_build_size_options
                    {27}, // log_probe_size_options
                    {1.0, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.0}, // hit_rate_options
                    {false, true}, // misses_are_outliers_options
                    {0}, // log_num_batches_options
                    {1.0}, // build_key_uniformity_options
                    {0.0}, // probe_zipf_coefficient_options
                    {0}, // log_key_multiplicity_options
                    {false}, // sort_insert_options
                    {false, true} // sort_probe_options
            },
            {
                    "batching",
                    {26}, // log_build_size_options
                    {27}, // log_probe_size_options
                    {1.0}, // hit_rate_options
                    {false}, // misses_are_outliers_options
                    {0, 4, 8, 12, 16, 20}, // log_num_batches_options
                    {1.0}, // build_key_uniformity_options
                    {0.0}, // probe_zipf_coefficient_options
                    {0}, // log_key_multiplicity_options
                    {false}, // sort_insert_options
                    {false, true} // sort_probe_options
            },
            {
                    "probe_skew",
                    {26}, // log_build_size_options
                    {27}, // log_probe_size_options
                    {1.0}, // hit_rate_options
                    {false}, // misses_are_outliers_options
                    {0}, // log_num_batches_options
                    {1.0}, // build_key_uniformity_options
                    {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0}, // probe_zipf_coefficient_options
                    {0}, // log_key_multiplicity_options
                    {false}, // sort_insert_options
                    {false, true} // sort_probe_options
            },
            {
                    "build_skew",
                    {26}, // log_build_size_options
                    {27}, // log_probe_size_options
                    {1.0}, // hit_rate_options
                    {false}, // misses_are_outliers_options
                    {0}, // log_num_batches_options
                    {1.0, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0}, // build_key_uniformity_options
                    {0.0}, // probe_zipf_coefficient_options
                    {0}, // log_key_multiplicity_options
                    {false}, // sort_insert_options
                    {false, true} // sort_probe_options
            },
            {
                    "key_multiplicity",
                    {26}, // log_build_size_options
                    {27}, // log_probe_size_options
                    {1.0}, // hit_rate_options
                    {false}, // misses_are_outliers_options
                    {0}, // log_num_batches_options
                    {1.0}, // build_key_uniformity_options
                    {0.0}, // probe_zipf_coefficient_options
                    {0, 1, 2, 3, 4, 5, 6, 7, 8}, // log_key_multiplicity_options
                    {false}, // sort_insert_options
                    {false, true} // sort_probe_options
            }
    };
    if (run_advanced_tests) {
        std::copy(
                advanced_test_configuration_options.begin(),
                advanced_test_configuration_options.end(),
                std::back_inserter(test_configuration_options)
        );
    }
#endif

    // pre-generate unique keys
    std::vector<key_type> unique_build_key_pool(max_build_size);
    {
        nvtx3::scoped_range_in<nvtx_benchmark_domain> pool{"key-pool-generation"};
        draw_without_replacement(unique_build_key_pool.begin(), unique_build_key_pool.size(), min_usable_key<key_type>, max_usable_key<key_type>);
    }
    std::cerr << "key pool generated" << std::endl;

    for (auto& tc : test_configuration_options) {
    for (auto log_build_size : tc.log_build_size_options) {
    for (auto log_probe_size : tc.log_probe_size_options) {
    for (auto hit_rate : tc.hit_rate_options) {
    for (auto misses_are_outliers : tc.misses_are_outliers_options) {
    for (auto log_num_batches : tc.log_num_batches_options) {
    for (auto build_key_uniformity : tc.build_key_uniformity_options) {
    for (auto probe_zipf_coefficient : tc.probe_zipf_coefficient_options) {
    for (auto log_key_multiplicity : tc.log_key_multiplicity_options) {
    for (auto sort_insert : tc.sort_insert_options) {
    for (auto sort_probe : tc.sort_probe_options) {

        std::stringstream experiment_description;
        {
            experiment_description << "pointq_" << index_type::short_description << "_" << tc.description << "_";
            experiment_description << sizeof(key_type) * 8 << "b_" << log_build_size << "/" << log_probe_size << "_";
            experiment_description << "batch" << log_num_batches << "_";
            experiment_description << (sort_insert ? "si" : "ui") << "_" << (sort_probe ? "sp" : "up") << "_";
            experiment_description << "hit" << hit_rate << (misses_are_outliers ? "!" : "") << "_";
            experiment_description << "uni" << build_key_uniformity << "_zipf" << probe_zipf_coefficient << "_";
            experiment_description << "mult" << log_key_multiplicity;
        }
        std::cerr << experiment_description.str() << std::endl;
        nvtx3::scoped_range_in<nvtx_benchmark_domain> experiment{"experiment-" + experiment_description.str()};

        // cannot have more batches than elements
        if (log_num_batches > log_probe_size)
            continue;

        size_t build_size = size_t{1} << log_build_size;
        size_t probe_size = size_t{1} << log_probe_size;
        size_t num_batches = size_t{1} << log_num_batches;
        size_t batch_size = size_t{1} << (log_probe_size - log_num_batches);
        size_t key_multiplicity = size_t{1} << log_key_multiplicity;

        // index does not support operation
        bool skip = key_multiplicity > 1 ? !index_type::can_multi_lookup : !index_type::can_lookup;
        if (skip) continue;

        // not enough memory
        if (free_memory < estimate_pq_memory_consumption<key_type, value_type>(build_size, probe_size))
            continue;

        rti_assert(batch_size * num_batches == probe_size);
        rti_assert(0 <= hit_rate && hit_rate <= 1);

        std::vector<key_type> build_keys, probe_keys;
        std::vector<value_type> build_values, expected_result;
        {
            nvtx3::scoped_range_in<nvtx_benchmark_domain> gen{"input-gen"};
            generate_point_query_input(
                build_size,
                probe_size,
                sort_insert,
                sort_probe,
                build_key_uniformity,
                probe_zipf_coefficient,
                hit_rate,
                misses_are_outliers,
                num_batches,
                batch_size,
                key_multiplicity,
                unique_build_key_pool,
                check_all_results,
                build_keys, build_values, probe_keys, expected_result);
        }

        cuda_buffer build_keys_buffer, build_values_buffer, probe_keys_buffer, result_buffer;
        build_keys_buffer.alloc_and_upload(build_keys);
        build_values_buffer.alloc_and_upload(build_values);
        probe_keys_buffer.alloc_and_upload(probe_keys);
        result_buffer.alloc(probe_size * sizeof(value_type));
        cudaMemset(result_buffer.raw_ptr, 0, result_buffer.size_in_bytes);

        double build_time_ms = 0, sort_time_ms = 0, probe_time_ms = 0;
        size_t build_bytes = 0, gpu_resident_bytes = 0;

        std::cerr << " setup complete" << std::endl;

        for (size_t run = 0; run < runs + 1; ++run) {
            // ignore first run due to weird CUDA behavior
            bool ignore = run == 0;

            nvtx3::scoped_range_in<nvtx_benchmark_domain> gen{"run-" + std::to_string(run)};

            index_type index;

            {
                nvtx3::scoped_range_in<nvtx_benchmark_domain> build{"build-phase"};
                index.build(build_keys_buffer.ptr<key_type>(), build_size, ignore ? nullptr : &build_time_ms, ignore ? nullptr : &build_bytes);
            }

            //std::cerr << "  build: " << build_time_ms << "ms" << std::endl;

            gpu_resident_bytes += ignore ? 0 : index.gpu_resident_bytes();

            // alloc sort buffers
            cuda_buffer sort_temp_buffer, sorted_probe_keys_buffer;
            size_t sort_temp_bytes = 0;
            if (sort_probe) {
                sort_temp_bytes = find_sort_buffer_size<key_type>(batch_size);
                sort_temp_buffer.alloc(sort_temp_bytes);
                sorted_probe_keys_buffer.alloc(batch_size * sizeof(key_type));
            }

            {
                nvtx3::scoped_range_in<nvtx_benchmark_domain> lookup{"lookup-phase"};
                cuda_timer timer(0);
                for (size_t batch = 0; batch < num_batches; ++batch) {
                    nvtx3::scoped_range_in<nvtx_benchmark_domain> batch_label{std::string("batch-") + std::to_string(batch)};

                    size_t offset = batch * batch_size;

                    key_type* probe_keys_batch_d = probe_keys_buffer.ptr<key_type>() + offset;
                    value_type* result_batch_d = result_buffer.ptr<value_type>() + offset;

                    if (sort_probe) {
                        nvtx3::scoped_range_in<nvtx_benchmark_domain> sort_batch{"sort-batch"};
                        key_type* sorted_probe_keys_d = sorted_probe_keys_buffer.ptr<key_type>();
                        timer.start();
                        untimed_sort(sort_temp_buffer.raw_ptr, sort_temp_bytes, probe_keys_batch_d, sorted_probe_keys_d, batch_size);
                        timer.stop();
                        probe_keys_batch_d = sorted_probe_keys_d;
                        sort_time_ms += ignore ? 0 : timer.time_ms();
                    }

                    {
                        nvtx3::scoped_range_in<nvtx_benchmark_domain> lookup_batch{"lookup-batch"};
                        timer.start();
                        if (key_multiplicity > 1) {
                            index.multi_lookup_sum(build_values_buffer.ptr<value_type>(), probe_keys_batch_d, result_batch_d, batch_size, 0);
                        } else {
                            index.lookup(build_values_buffer.ptr<value_type>(), probe_keys_batch_d, result_batch_d, batch_size, 0);
                        }
                        timer.stop();
                        probe_time_ms += ignore ? 0 : timer.time_ms();
                    }
                    cudaDeviceSynchronize(); CUERR
                }
            }

            std::cerr << "  run " << run << " complete" << (ignore ? " (ignored)" : "") << std::endl;

            if (check_all_results) check_result(probe_keys, probe_keys, expected_result, result_buffer);
        }

        std::cerr << " -> " << (probe_time_ms / runs) << " ms" << std::endl;

        rc.add("i_configuration", tc.description);
        rc.add("i_type", index_type::short_description);
        rc.add("i_runs", runs);
        rc.add("i_key_size", sizeof(key_type) * 8);
        rc.add("i_value_size", sizeof(value_type) * 8);
        rc.add("i_log_build_size", log_build_size);
        rc.add("i_log_probe_size", log_probe_size);
        rc.add("i_log_num_batches", log_num_batches);
        rc.add("i_sort_insert", sort_insert);
        rc.add("i_sort_probe", sort_probe);
        rc.add("i_hit_rate", hit_rate);
        rc.add("i_misses_are_outliers", misses_are_outliers);
        rc.add("i_build_key_uniformity", build_key_uniformity);
        rc.add("i_probe_zipf_coefficient", probe_zipf_coefficient);
        rc.add("i_log_key_multiplicity", log_key_multiplicity);

        rc.add("checked_result", check_all_results);
        rc.add("build_time_ms", build_time_ms / runs);
        rc.add("sort_time_ms", sort_time_ms / runs);
        rc.add("probe_time_ms", probe_time_ms / runs);
        rc.add("build_bytes", build_bytes / runs);
        rc.add("gpu_resident_bytes", gpu_resident_bytes / runs);
        rc.commit_line();
    }}}}}}}}}}}
}


struct range_query_configuration {
    size_t log_build_size;
    size_t log_probe_size;
    size_t log_key_range_factor;
    size_t log_range_size;
    size_t log_num_batches;
};


template <typename index_type, typename value_type>
void benchmark_range_query(
        rc::result_collector& rc,
        size_t runs,
        size_t max_log_build_size,
        size_t max_log_probe_size,
        bool check_all_results
) {

    nvtx3::scoped_range_in<nvtx_benchmark_domain> index_experiment{index_type::short_description};

    using key_type = typename index_type::key_type;

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

#if 0
#warning "a test setup is configured, pre-configured benchmarks will be skipped"
    std::vector<range_query_configuration> configuration_options { {26, 27, 0, 6, 0} };
    std::vector<bool> sort_insert_options {false};
    std::vector<bool> sort_probe_options {false};
#else
    std::vector<range_query_configuration> configuration_options {
        // dense keys
        {26, 27, 0,  0, 0},
        {26, 27, 0,  2, 0},
        {26, 27, 0,  4, 0},
        {26, 27, 0,  6, 0},
        {26, 27, 0,  8, 0},
        {26, 27, 0, 10, 0},
        // sparse keys
        {26, 27,  2, 10, 0},
        {26, 27,  4, 10, 0},
        {26, 27,  6, 10, 0},
        {26, 27,  8, 10, 0},
        {26, 27, 10, 10, 0},
        // smaller build size
        {25, 27, 6, 10, 0},
        {24, 27, 6, 10, 0},
        {23, 27, 6, 10, 0},
        {22, 27, 6, 10, 0},
        {21, 27, 6, 10, 0},
        {20, 27, 6, 10, 0},
        {19, 27, 6, 10, 0}
    };
    std::vector<bool> sort_insert_options {false};
    std::vector<bool> sort_probe_options {false, true};
#endif

    for (auto configuration : configuration_options) {
    for (bool sort_insert : sort_insert_options) {
    for (bool sort_probe : sort_probe_options) {

        size_t log_build_size = configuration.log_build_size;
        size_t log_probe_size = configuration.log_probe_size;
        size_t log_key_range_factor = configuration.log_key_range_factor;
        size_t log_range_size = configuration.log_range_size;
        size_t log_num_batches = configuration.log_num_batches;

        std::stringstream experiment_description;
        {
            experiment_description << "rangeq_" << index_type::short_description << "_";
            experiment_description << sizeof(key_type) * 8 << "b_" << log_build_size << "/" << log_probe_size << "_";
            experiment_description << "batch" << log_num_batches << "_";
            experiment_description << "range" << log_range_size << "_factor" << log_key_range_factor << "_";
            experiment_description << (sort_insert ? "si" : "ui") << "_" << (sort_probe ? "sp" : "up");
        }
        std::cerr << experiment_description.str() << std::endl;
        nvtx3::scoped_range_in<nvtx_benchmark_domain> experiment{"experiment-" + experiment_description.str()};

        // cannot have more batches than elements
        if (log_num_batches > log_probe_size)
            continue;
        if (log_build_size > max_log_build_size)
            continue;
        if (log_probe_size > max_log_probe_size)
            continue;

        size_t build_size = size_t{1} << log_build_size;
        size_t probe_size = size_t{1} << log_probe_size;
        key_type range_size = key_type{1} << log_range_size;
        size_t num_batches = size_t{1} << log_num_batches;
        size_t batch_size = size_t{1} << (log_probe_size - log_num_batches);
        size_t key_range_factor = size_t{1} << log_key_range_factor;
        size_t key_range = build_size * key_range_factor;

        // make sure all keys can be represented
        if (std::log2(key_range) >= 8 * sizeof(key_type))
            continue;
        // make sure there are enough keys to allow a range query of the specified size (with added margin)
        if (range_size * 2 > key_range)
            continue;
        if (free_memory < estimate_rq_memory_consumption<key_type, value_type>(build_size, probe_size))
            continue;

        std::vector<key_type> build_keys, lower_keys, upper_keys;
        std::vector<value_type> build_values, expected_result;
        {
            nvtx3::scoped_range_in<nvtx_benchmark_domain> gen{"input-gen"};
            generate_range_query_input(
                    build_size, probe_size, sort_insert, sort_probe, range_size, (key_type) key_range, num_batches, batch_size, check_all_results,
                    build_keys, build_values, lower_keys, upper_keys, expected_result);
        }

        cuda_buffer build_keys_buffer, build_values_buffer, lower_keys_buffer, upper_keys_buffer, result_buffer;
        build_keys_buffer.alloc_and_upload(build_keys);
        build_values_buffer.alloc_and_upload(build_values);
        lower_keys_buffer.alloc_and_upload(lower_keys);
        upper_keys_buffer.alloc_and_upload(upper_keys);
        result_buffer.alloc(probe_size * sizeof(value_type));
        cudaMemset(result_buffer.raw_ptr, 0, result_buffer.size_in_bytes);

        double build_time_ms = 0, sort_time_ms = 0, probe_time_ms = 0;
        size_t build_bytes = 0, gpu_resident_bytes = 0;

        std::cerr << " setup complete" << std::endl;

        for (size_t run = 0; run < runs + 1; ++run) {
            // ignore first run due to weird optix behavior
            bool ignore = run == 0;

            nvtx3::scoped_range_in<nvtx_benchmark_domain> gen{"run-" + std::to_string(run)};

            index_type index;

            {
                nvtx3::scoped_range_in<nvtx_benchmark_domain> build{"build-phase"};
                index.build(build_keys_buffer.ptr<key_type>(), build_size, ignore ? nullptr : &build_time_ms, ignore ? nullptr : &build_bytes);
            }

            gpu_resident_bytes += ignore ? 0 : index.gpu_resident_bytes();

            // alloc sort buffers
            cuda_buffer sort_temp_buffer, sorted_lower_keys_buffer, sorted_upper_keys_buffer;
            size_t sort_temp_bytes = 0;
            if (sort_probe) {
                sort_temp_bytes = find_pair_sort_buffer_size<key_type, key_type>(batch_size);
                sort_temp_buffer.alloc(sort_temp_bytes);
                sorted_lower_keys_buffer.alloc(batch_size * sizeof(key_type));
                sorted_upper_keys_buffer.alloc(batch_size * sizeof(key_type));
            }

            {
                nvtx3::scoped_range_in<nvtx_benchmark_domain> lookup{"lookup-phase"};
                cuda_timer timer(0);
                for (size_t batch = 0; batch < num_batches; ++batch) {
                    nvtx3::scoped_range_in<nvtx_benchmark_domain> batch_label{std::string("batch-") + std::to_string(batch)};

                    size_t offset = batch * batch_size;

                    key_type* probe_lower_batch_d = lower_keys_buffer.ptr<key_type>() + offset;
                    key_type* probe_upper_batch_d = upper_keys_buffer.ptr<key_type>() + offset;
                    value_type* result_batch_d = result_buffer.ptr<value_type>() + offset;

                    if (sort_probe) {
                        nvtx3::scoped_range_in<nvtx_benchmark_domain> sort_batch{"sort-batch"};
                        key_type* sorted_probe_lower_d = sorted_lower_keys_buffer.ptr<key_type>();
                        key_type* sorted_probe_upper_d = sorted_upper_keys_buffer.ptr<key_type>();
                        timer.start();
                        untimed_pair_sort(sort_temp_buffer.raw_ptr, sort_temp_bytes, probe_lower_batch_d, sorted_probe_lower_d, probe_upper_batch_d, sorted_probe_upper_d, batch_size);
                        timer.stop();
                        probe_lower_batch_d = sorted_probe_lower_d;
                        probe_upper_batch_d = sorted_probe_upper_d;
                        sort_time_ms += ignore ? 0 : timer.time_ms();
                    }
                    {
                        nvtx3::scoped_range_in<nvtx_benchmark_domain> lookup_batch{"lookup-batch"};
                        timer.start();
                        index.range_lookup_sum(build_values_buffer.ptr<value_type>(), probe_lower_batch_d, probe_upper_batch_d, result_batch_d, batch_size, 0);
                        timer.stop();
                        cudaDeviceSynchronize(); CUERR
                        probe_time_ms += ignore ? 0 : timer.time_ms();
                    }
                }
            }

            if (check_all_results) check_result(lower_keys, upper_keys, expected_result, result_buffer);
        }

        std::cerr << " -> " << (probe_time_ms / runs) << " ms" << std::endl;

        rc.add("i_type", index_type::short_description);
        rc.add("i_runs", runs);
        rc.add("i_key_size", sizeof(key_type) * 8);
        rc.add("i_value_size", sizeof(value_type) * 8);
        rc.add("i_log_build_size", log_build_size);
        rc.add("i_log_probe_size", log_probe_size);
        rc.add("i_log_range_size", log_range_size);
        rc.add("i_log_num_batches", log_num_batches);
        rc.add("i_sort_insert", sort_insert);
        rc.add("i_sort_probe", sort_probe);
        rc.add("i_log_key_range_factor", log_key_range_factor);
        rc.add("i_key_range", key_range);

        rc.add("checked_result", check_all_results);
        rc.add("build_time_ms", build_time_ms / runs);
        rc.add("sort_time_ms", sort_time_ms / runs);
        rc.add("probe_time_ms", probe_time_ms / runs);
        rc.add("build_bytes", build_bytes / runs);
        rc.add("gpu_resident_bytes", gpu_resident_bytes / runs);
        rc.commit_line();
    }}}
}


template <typename index_type, typename value_type>
void benchmark_range_query(rc::result_collector& rc, size_t runs, bool check_all_results) {
    benchmark_range_query<index_type, value_type>(rc, runs, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), check_all_results);
}

#endif
