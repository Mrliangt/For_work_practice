// test_global_lru.cc
// GlobalLRU_cache 的验证程序:单线程正确性 + 多线程压力(值一致性)。
#include "LRU_cache.h"

#include <atomic>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

void test_basic() {
    GlobalLRU_cache<int, std::string> cache(3);

    cache.Put(1, "a");
    cache.Put(2, "b");
    cache.Put(3, "c");
    assert(cache.Size() == 3);

    assert(cache.Get(1) && *cache.Get(1) == "a");  // 刚插入,必命中
    assert(cache.Contains(1) && cache.Contains(2) && cache.Contains(3));

    cache.Put(1, "aa");                            // 原地更新
    assert(cache.Get(1) && *cache.Get(1) == "aa");
    assert(cache.Size() == 3);

    assert(cache.Erase(2));
    assert(!cache.Contains(2));
    assert(!cache.Erase(2));                       // 重复删除返回 false
    assert(cache.Size() == 2);

    cache.Clear();
    assert(cache.Size() == 0);
    assert(!cache.Get(1));
}

void test_capacity_bounded() {
    // CLOCK 不是严格 LRU,不能断言"谁被淘汰";但容量上界、
    // 以及"存活 key 读到的值必与其 key 一致"是确定的。
    GlobalLRU_cache<int, std::string> cache(16);
    for (int i = 0; i < 1000; ++i)
        cache.Put(i, std::to_string(i));

    assert(cache.Size() <= 16);

    for (int i = 0; i < 1000; ++i) {
        if (auto v = cache.Get(i))
            assert(*v == std::to_string(i));
    }
}

void test_second_chance() {
    // 引用位语义:被访问过的条目应比从未访问过的条目存活更久。
    // 下面这个序列在 CLOCK 下是确定的(hand 从 0 开始,新增条目引用位为 0):
    //   Put 1,2,3 → Get(1) 置引用位 → Put(4) 必须淘汰一个 →
    //   时针先经过 1(有引用位,放行)、再经过 2(无引用位,牺牲)。
    GlobalLRU_cache<int, std::string> cache(3);
    cache.Put(1, "a");
    cache.Put(2, "b");
    cache.Put(3, "c");
    assert(cache.Get(1));
    cache.Put(4, "d");

    assert(cache.Contains(1));   // 被访问 → 存活
    assert(!cache.Contains(2));  // 未访问 → 牺牲
    assert(cache.Contains(3) && cache.Contains(4));
    assert(cache.Size() == 3);
}

void test_concurrent() {
    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 200000;

    GlobalLRU_cache<std::string, std::string> cache(1024, 32);

    std::atomic<std::uint64_t> errors{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937 rng(static_cast<unsigned>(t + 1));
            std::uniform_int_distribution<int> dist(0, 8191);
            for (int i = 0; i < kOpsPerThread; ++i) {
                const std::string key = std::to_string(dist(rng));
                if (dist(rng) % 4 == 0) {   // 25% 写
                    cache.Put(key, key);
                } else {                     // 75% 读
                    auto v = cache.Get(key);
                    // 不变式:凡非空返回值,必须与 key 一致。
                    // 这能抓住 use-after-free / 错读他人槽位 / 撕裂读。
                    if (v && *v != key)
                        errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads)
        th.join();

    assert(errors.load() == 0);
    assert(cache.Size() <= 1024);

    std::cout << "  concurrent: hits=" << cache.Hits()
              << " misses=" << cache.Misses()
              << " hit_rate=" << cache.HitRate() << '\n';
}

}  // namespace

int main() {
    std::cout << "== GlobalLRU_cache tests ==\n";
    test_basic();
    std::cout << "  basic: ok\n";
    test_capacity_bounded();
    std::cout << "  capacity: ok\n";
    test_second_chance();
    std::cout << "  second-chance: ok\n";
    test_concurrent();
    std::cout << "  concurrent: ok\n";
    std::cout << "All tests passed.\n";
}
