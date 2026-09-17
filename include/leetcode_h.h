/**
 * @file leetcode_h
 * @author tian
 * @date 2026/8/28
 */
#pragma once

#include <vector>
#include <iostream>
#include <unordered_set>
#include <stack>
#include <queue>
#include <sstream>
#include <string>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int val) : val(val), left(nullptr), right(nullptr) {}
    TreeNode(int val, TreeNode* left, TreeNode* right) : val(val), left(left), right(right) {}
};

struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int v) : val(v), next(nullptr) {}
    ListNode(int v, ListNode* n) : val(v), next(n) {}
};

class Solution {
public:
    /* 162.寻找峰值  --思路：二分；
     * 原理：如果因为定义了nums[-1]和nums[size] 为负无穷
     * 所有如果nums[i] > nums[i + 1]则[0,i]中一定有峰值
     * */
    int findPeakElement(std::vector<int>& nums) {
        int size = nums.size();
        int left = 0, right = size - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    /* 852.山脉数组的峰值索引
     * 原理基本同上，思考一下
     * */
    int peakIndexInMountainArray(std::vector<int>& arr) {
        int size = arr.size();
        int left = 0, right = size - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] > arr[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    /* 113.路径总和Ⅱ
     * 回溯
     * */
    std::vector<std::vector<int>> pathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) {
            return {};
        }
        path.clear();
        res.clear();
        path.push_back(root->val);
        findPath(root, targetSum - root->val);
        return res;
    }
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    void findPath(TreeNode* root, int targetSum) {
        if (root->left == nullptr && root->right == nullptr && targetSum == 0) {
            res.push_back(path);
            return;
        }
        if (root->left != nullptr) {
            path.push_back(root->left->val);
            findPath(root->left, targetSum - root->left->val);
            path.pop_back();
        }

        if (root->right != nullptr) {
            path.push_back(root->right->val);
            findPath(root->right, targetSum - root->right->val);
            path.pop_back();
        }
    }

    /* 62.不同路径
     *  动态规划
     * */
    int uniquePaths(int m, int n) {
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
        dp[0][1] = 1;
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m][n];
    }

    /* 24.两两交换链表中的节点
     *  虚拟头节点
     * */
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }

        ListNode* dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode* pre = dummyNode, *cur = head, *neibour = cur->next;
        while (neibour != nullptr) {
            pre->next = neibour;
            cur->next = neibour->next;
            neibour->next = cur;

            pre = cur;
            cur = cur->next;
            if (cur == nullptr) {
                break;
            }
            neibour = cur->next;
        }
    }

    /* 112. 路径总和 easy
     * 递归
     * */
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) {
            return false;
        }
        if (root->left == nullptr && root->right == nullptr && root->val == targetSum) {
            return true;
        }

        return hasPathSum(root->left, targetSum - root->val)
                || hasPathSum(root->right, targetSum - root->val);
    }

    /* 139.单词拆分
     * dp[i] = s的前i个字符，能否被字典中的单词拼出来
     * */
    bool wordBreak(std::string s, std::vector<std::string>& wordDict) {
        std::unordered_set<std::string> dict(wordDict.begin(), wordDict.end());

        int size = s.size();
        std::vector<int> dp(size + 1, 0);
        dp[0] = true;

        for (int i = 1; i <= size; ++i) {
            for (int j = 0; j < i; ++j) {
                if (dp[j] && dict.find(s.substr(j, i - j)) != dict.end()) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[size];
    }

    /* 131.分割回文串
     * 回溯
     * */
    std::vector<std::vector<std::string>> partition(std::string s) {
        std::vector<std::string> path;
        std::vector<std::vector<std::string>> res;

        auto isPalindrome = [&](int start, int end) {
            while (start <= end) {
                if (s[start] != s[end]) {
                    return false;
                }
                start++;
                end--;
            }
            return true;
        };

        auto dfs = [&] (auto&& dfs, int start) {
            if (start == s.size()) {
                res.push_back(path);
                return;
            }

            for (int end = start; end < s.size(); ++end) {
                if (!isPalindrome(start, end)) {
                    continue;
                }

                path.push_back(s.substr(start, end - start + 1));
                dfs(dfs, end + 1);
                path.pop_back();
            }
        };

        dfs(dfs, 0);
        return res;
    }

    /* 718.最长重复子数组
     *  DP + 表格法
     * */
    int findLength(std::vector<int>& nums1, std::vector<int>& nums2) {
        int size1 = nums1.size();
        int size2 = nums2.size();

        std::vector<std::vector<int>> dp(size1, std::vector<int>(size2, 0));
        int res = 0;
        for (int i = 0; i < size1; ++i) {
            int val1 = nums1[i];
            for (int j = 0; j < size2; ++j) {
                int val2 = nums2[j];
                if (val1 == val2) {
                    dp[i][j] = 1;
                    if (i > 0 && j > 0) {
                        dp[i][j] += dp[i - 1][j - 1];
                    }
                    res = std::max(res, dp[i][j]);
                }
            }
        }
        return res;
    }

    /* 83.删除排序链表中的重复元素
     *
     * */
    ListNode* deleteDuplicates(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }

        ListNode* cur = head;
        while (cur->next != nullptr) {
            if (cur->val == cur->next->val) {
                cur->next = cur->next->next;
            } else {
                cur = cur->next;
            }
        }
        return head;
    }

    /* 227.基本计算器Ⅱ
     * 栈 & 字符串处理
     * */
    int calculate(std::string s) {
        auto cal = [&] (auto&& cal, int& begin) -> int {
            long long res = 0;
            std::stack<int> nums;
            char ops = '+';

            auto take = [&](long long num, char ops) {
                long long pre = 0;
                switch (ops) {
                    case '+': nums.push(num); break;
                    case '-': nums.push(-num); break;
                    case '*': pre = nums.top(); nums.pop(); nums.push(pre * num); break;
                    case '/': pre = nums.top(); nums.pop(); nums.push(pre / num); break;
                }
            };

            long long num = 0;

            while (begin < s.size()) {
                char c = s[begin];

                if (c >= '0' && c <= '9') {
                    num = num * 10 + c - '0';
                    ++begin;
                } else if (c == ' ') {
                    ++begin;
                    continue;
                } else if (c == '(') {
                    ++begin;
                    num = cal(cal, begin);
                } else if (c == ')') {
                    take(num, ops);
                    ++begin;
                    long long res = 0;
                    while (!nums.empty()) {
                        res += nums.top();
                        nums.pop();
                    }
                    return res;
                } else {
                    take(num, ops);
                    ++begin;
                    ops = c;
                    num = 0;
                }
            }
            take(num, ops);

            while (!nums.empty()) {
                res += nums.top();
                nums.pop();
            }
            return res;
        };

        int begin = 0;
        return cal(cal, begin);
    }

    /* 226.翻转二叉树
     * 递归
     * */
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) {
            return root;
        }

        TreeNode* temp = root->left;
        root->left = root->right;
        root->right = temp;

        invertTree(root->left);
        invertTree(root->right);
        return root;
    }

    /* 169.多数元素
     * 摩尔投票
     * */
    int majorityElement(std::vector<int>& nums) {
        int vote = 0, num = 0;
        for (auto n : nums) {
            if (vote == 0) {
                num = n;
            }
            vote += (n == num) ? 1 : -1;
        }
        return num;
    }

    /* 207.课程表
     * BFS + 入度 + 队列存可上的课
     * */
    bool canFinish(int numCourses, std::vector<std::vector<int>>& prerequisites) {
        std::vector<std::vector<int>> graph(numCourses); // 构建图的邻接表
        std::vector<int> indegree(numCourses);           // 入度统计数组

        int isfinished = 0;    // 统计完成课程

        // 构建邻接表
        for (const auto& pre : prerequisites) {
            int course = pre[0];
            int prerequisite = pre[1];

            graph[prerequisite].push_back(course);
            indegree[course]++;
        }

        // 放入度为0的课程到就绪队列中
        std::queue<int> que;

        for (int i = 0; i < numCourses; ++i) {
            if (indegree[i] == 0) {
                que.push(i);
            }
        }

        // 通过对入度操作不断更新就绪队列
        while (!que.empty()) {
            int course = que.front();
            que.pop();

            isfinished++;

            for (auto& nextCourse : graph[course]) {
                indegree[nextCourse]--;

                if (indegree[nextCourse] == 0) {
                    que.push(nextCourse);
                }
            }
        }
        return isfinished == numCourses;
    }

    /* 283. 移动零
     *  双下标
     * */
    void moveZeroes(std::vector<int>& nums) {
        int index = 0;
        int pos = 0;
        for (index = 0; index < nums.size() - 1; ++index) {
            if (nums[index] == 0) {
                while (pos < nums.size() - 1 && nums[pos] == 0) {
                    pos++;
                }
                nums[index] = nums[pos];
                nums[pos] = 0;
            } else {
                pos++;
            }
        }
    }

    /* 滴滴笔试题---合影
     * description:学校组织了集体合影，N个同学站在N级台阶上，学号从1-n,每个台阶高度为D,每一级台阶必须有人,pi表示站
     * 在从下往上的第i个台阶上的同学的学号,每个同学都有一个身高值,学号为i的同学的身高用Ai表示,为了防止遮挡,对于所有
     * 都要满足Api+1 >= Api - D,计算一共有多少种合法的排列方式。
     * 树状数组
     * */
    class Fenwick {
    private:
        int n;
        std::vector<int> tree;
    public:
        Fenwick(int n) : n(n), tree(n + 1, 0) {}

        void add(int index, int value) {
            while (index <= n) {
                tree[index] += value;
                index += index & -index;
            }
        }

        int sum(int index) const {
            int result = 0;

            while (index > 0) {
                result += tree[index];
                index -= index & -index;
            }
            return result;
        }
    };
    using ll = long long;
    int countWays(std::vector<ll>& heights, ll D) {
        static constexpr ll MOD = 998244353;
        int n = heights.size();

        // 按身高从高到低
        std::vector<ll> order = heights;
        std::sort(order.begin(), order.end(), std::greater<ll>());

        // 离散化数组
        std::vector<ll> values = heights;
        std::sort(values.begin(), values.end());

        values.erase(std::unique(values.begin(), values.end()), values.end());

        // 创建树状数组
        Fenwick bit(values.size());

        ll answer = 1;

        for (ll x : order) {
            ll limit = x + D;

            int rankCount = std::upper_bound(values.begin(), values.end(), limit) - values.begin();

            int validPrev = bit.sum(rankCount);

            answer = answer * (validPrev + 1) % MOD;

            int rank = std::lower_bound(values.begin(), values.end(), x) - values.begin() + 1;
            bit.add(rank, 1);
        }

        return static_cast<int>(answer);
    }

    /* 912. 排序数组
     * 分别实现快速排序 + 堆排序
     * */
    std::vector<int> sortArray(std::vector<int>& nums) {
         int size = nums.size();

         auto partition = [&] (int left, int right) -> int {
             int mid = left + (right - left) / 2;
             int val = nums[mid];
             nums[mid] = nums[left];

             while (left < right) {
                 while (left < right && nums[right] >= val) {
                     right--;
                 }
                 if (left < right) {
                     nums[left] = nums[right];
                 }
                 while (left < right && nums[left] <= val) {
                     left++;
                 }
                 if (left < right) {
                     nums[right] = nums[left];
                 }
             }
             nums[left] = val;
             return left;
         };

         auto quickSort = [&] (auto&& quickSort, int left, int right) {
             if (left >= right) {
                 return;
             }
             int pos = partition(left, right);

             quickSort(quickSort, left, pos - 1);
             quickSort(quickSort, pos + 1, right);
         };

         quickSort(quickSort, 0, size - 1);
         return nums;
    }
    std::vector<int> sortArray2(std::vector<int>& nums) {
        int size = nums.size();

        auto heapify = [&] (int heap_size, int root_index) {
            int largest = root_index;

            while (true) {
                int left_index = 2 * root_index + 1;
                int right_index = 2 * root_index + 2;

                if (left_index < heap_size && nums[left_index] > nums[largest]) {
                    largest = left_index;
                }

                if (right_index < heap_size && nums[right_index] > nums[largest]) {
                    largest = right_index;
                }

                if (largest != root_index) {
                    std::swap(nums[root_index], nums[largest]);
                    root_index = largest;
                } else {
                    break;
                }
            }
        };

        for (int i = size / 2 - 1; i >= 0; --i) {
            heapify(size, i);
        }

        for (int i = size - 1; i > 0; --i) {
            std::swap(nums[i], nums[0]);
            heapify(i, 0);
        }
        return nums;
    }

    /* 739. 每日温度
     * 单调栈
     * */
    std::vector<int> dailyTemperatures(std::vector<int>& temperatures) {
        int size = temperatures.size();
        std::vector<int> ans(size, 0);
        std::stack<int> stk;
        stk.push(0);

        for (int i = 1; i < size; ++i) {
            if (temperatures[i] < temperatures[stk.top()]) {
                stk.push(i);
            } else {
                while (!stk.empty() && temperatures[i] > temperatures[stk.top()]) {
                    ans[stk.top()] = i - stk.top();
                    stk.pop();
                }
                stk.push(i);
            }
        }
        return ans;
    }

    /* 468.验证IP地址
     * 纯字符串操作
     * */
    std::string validIPAddress(std::string queryIP) {
        auto split = [&] (std::string IP, char c, std::vector<std::string>& addr) {
            std::stringstream ss(IP);
            std::string ip;

            while (std::getline(ss, ip, c)) {
                addr.push_back(ip);
            }
            if (IP.size() > 0 && IP.back() == c) {
                addr.push_back({});
            }
        };

        auto isIPv4 = [&] () -> bool {
            std::vector<std::string> addr;
            split(queryIP, '.', addr);
            if (addr.size() != 4) {
                return false;
            }

            for (auto s : addr) {
                if (s.size() < 1 || (s.size() > 1 && s[0] == '0') || s.size() > 3) {
                    return false;
                }
                for (auto c : s) {
                    if (!isdigit(c)) {
                        return false;
                    }
                }

                int digit = std::stoi(s);
                if (digit < 0 || digit > 255) {
                    return false;
                }
            }
            return true;
        };

        auto isIPv6 = [&] () -> bool {
            std::vector<std::string> addr;
            split(queryIP, ':', addr);
            if (addr.size() != 8) {
                return false;
            }

            for (auto s : addr) {
                if (s.size() < 1 || s.size() > 4) {
                    return false;
                }

                for (auto c : s) {
                    if ((c < '0' || c > '9') && (c < 'a' || c > 'f') && (c < 'A' || c > 'F')) {
                        return false;
                    }
                }
            }
            return true;
        };

        if (isIPv4()) {
            return "IPv4";
        }
        if (isIPv6()) {
            return "IPv6";
        }
        return "Neither";
    }

    /* 79.单词搜索
     * DFS + 回溯
     * */
    bool exist(std::vector<std::vector<char>>& board, std::string word) {
        int rows = board.size();
        int cols = board[0].size();

        auto dfs = [&] (auto&& dfs, int i, int j, int k) ->bool {
            if (i < 0 || j < 0 || i >= rows || j >= cols || board[i][j] != word[k]) {
                return false;
            }
            if (k == word.size() - 1) {
                return true;
            }

            board[i][j] = '\0';
            bool res = dfs(dfs, i + 1, j, k + 1) || dfs(dfs, i - 1, j, k + 1)
                        || dfs(dfs, i, j + 1, k + 1) || dfs(dfs, i, j - 1, k + 1);
            board[i][j] = word[k];
            return res;
        };
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (dfs(dfs, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }
};












