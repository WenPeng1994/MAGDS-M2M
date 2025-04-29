% 示例数组
% A = [2,3,0,0];


A = [10,30,30,  9,  9,  3,  0,  0,  0,  0,  99, 9,  0, 0,  0];
% 计算方巧
variance = var(A)

% 为了归一化，假设最大方巧是数组极差的平方除以数组长度
% maxVariance = (range(A)^2) / length(A);
maxVariance = (range(A)^2);

% 将方巧映射到 0-1 范围
normalizedVariance = variance / (maxVariance+1e-8)