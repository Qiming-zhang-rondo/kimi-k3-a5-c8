// Proposed experimental ABI, not an existing exported vendor symbol.
// Requires generated/matching implementation and registration in a separate
// experimental library. The original aclnnSparseFlashAttention is unchanged.
#pragma once
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnSparseFlashAttentionColdMergeGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseIndices, const aclTensor *blockTable,
    const aclTensor *actualSeqLengthsQuery, const aclTensor *actualSeqLengthsKv,
    const aclTensor *queryRope, const aclTensor *keyRope,
    const aclTensor *coldOutput, const aclTensor *coldMax, const aclTensor *coldSum,
    const aclTensor *coldPresent, const aclTensor *hotPresent,
    double scaleValue, int64_t sparseBlockSize, char *layoutQuery, char *layoutKv,
    int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t attentionMode, bool returnSoftmaxLse,
    const aclTensor *attentionOut, const aclTensor *hotMax, const aclTensor *hotSum,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnSparseFlashAttentionColdMerge(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif
