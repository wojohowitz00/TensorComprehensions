/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/core/polyhedral/codegen_cuda.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/mapped_scop.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/memory_promotion_heuristic.h"
#include "tc/core/polyhedral/scop.h"

using namespace std;

using namespace tc;
using namespace tc::polyhedral;

namespace {
int npoints(isl::set s) {
  int cnt = 0;
  isl::union_set(s).foreach_point([&cnt](isl::point pt) { ++cnt; });
  return cnt;
}

isl::set projectOutThreadParams(isl::set s) {
  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  std::vector<isl::id> ids{TX, TY, TZ};
  for (auto id : ids) {
    int pos = s.find_dim_by_id(isl::dim_type::param, id);
    if (pos != -1) {
      s = s.project_out(isl::dim_type::param, pos, 1);
    }
  }
  return s;
}
} // namespace

class TestMapper : public ::testing::Test {
 protected:
  std::unique_ptr<MappedScop> makeMappedScop(
      std::string tc,
      const MappingOptions& mappingOptions,
      std::unordered_map<std::string, size_t> problemSizes) {
    auto ctx = isl::with_exceptions::globalIslCtx();
    auto scop = Scop::makeScop(ctx, tc);
    scop = Scop::makeSpecializedScop(
        *scop,
        scop->makeContext(problemSizes)
            .intersect(scop->globalParameterContext));
    scop->domain() =
        scop->domain().intersect_params(scop->globalParameterContext);
    return MappedScop::makeWithOuterBlockInnerThreadStrategy(
        std::move(scop), mappingOptions);
  }
};

class MapperMemoryPromotion2DHelper : public TestMapper {
 protected:
  std::unique_ptr<MappedScop> prepareScop(
      std::string tc,
      std::unordered_map<std::string, size_t> problemSizes,
      std::vector<size_t> tileSizes) {
    auto mappingOptions =
        MappingOptions::makeNaiveMappingOptions()
            .tile(tileSizes) // passing more tiling values triggers a check...
            .useSharedMemory(false) // do not auto-promote
            .usePrivateMemory(true);
    return makeMappedScop(tc, mappingOptions, problemSizes);
  }
};

class Sum4D : public TestMapper {
 public:
  std::string emitCode(
      std::vector<size_t> problemSizes,
      std::vector<size_t> tileSizes,
      std::vector<size_t> childPos) {
    string tc = R"TC(
def fun(float(N,M,K,L) A, float(N,M,K,L) B) -> (C) {
  C(i,j,k,l) = A(i,j,k,l) + B(i,j,k,l)
}
)TC";

    auto mappingOptions = MappingOptions::makeNaiveMappingOptions()
                              .tile(tileSizes)
                              .useSharedMemory(false) // do not autopromote
                              .usePrivateMemory(true);
    auto mscop = makeMappedScop(
        tc,
        mappingOptions,
        std::unordered_map<string, size_t>{{"N", problemSizes[0]},
                                           {"M", problemSizes[1]},
                                           {"K", problemSizes[2]},
                                           {"L", problemSizes[3]}});
    auto& scop = const_cast<Scop&>(mscop->scop());
    scop.promoteEverythingAt(childPos);
    return std::get<0>(mscop->codegen("fun"));
  }
};

TEST_F(Sum4D, CodeOuterBand) {
  auto declarations = {"__shared__ float32 _A_0[16][16][16][16];",
                       "__shared__ float32 _B_0[16][16][16][16];",
                       "__shared__ float32 _C_0[16][16][16][16];"};

  auto copyA =
      "_A_0[c4][c5][c6][c7] = A[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7];";
  auto copyB =
      "_B_0[c4][c5][c6][c7] = B[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7];";
  auto compute =
      "_C_0[c4][c5][c6][t0] = (_A_0[c4][c5][c6][t0] + _B_0[c4][c5][c6][t0]);";
  auto copyC =
      "C[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7] = _C_0[c4][c5][c6][c7];";
  auto sync = "__syncthreads()";

  auto code = emitCode({256, 128, 192, 224}, {16, 16, 16, 16}, {0, 0, 0, 0});
  // Order of copies may be arbitrary, but syncs must be inserted before and
  // after
  for (auto d : declarations) {
    EXPECT_TRUE(code.find(d) != std::string::npos);
  }
  auto posA = code.find(copyA);
  auto posB = code.find(copyB);
  auto posCompute = code.find(compute);
  auto posC = code.find(copyC);
  auto posSync1 = code.find(sync);
  auto posSync2 = code.rfind(sync, posCompute);
  auto posSync3 = code.find(sync, posCompute);
  auto posSync4 = code.rfind(sync);
  EXPECT_NE(posA, std::string::npos);
  EXPECT_NE(posB, std::string::npos);
  EXPECT_NE(posC, std::string::npos);
  EXPECT_NE(posCompute, std::string::npos);
  EXPECT_NE(posSync1, std::string::npos);
  EXPECT_NE(posSync2, std::string::npos);
  EXPECT_NE(posSync3, std::string::npos);
  EXPECT_NE(posSync4, std::string::npos);
  EXPECT_LT(posSync1, std::min(posA, posB));
  EXPECT_GT(posSync2, std::max(posA, posB));
  EXPECT_LT(posSync3, posC);
  EXPECT_GT(posSync4, posC);
}

TEST_F(Sum4D, CodeBeforeThreadMapping) {
  auto declarations = {"__shared__ float32 _A_0[16][16][16][1];",
                       "__shared__ float32 _B_0[16][16][16][1];",
                       "__shared__ float32 _C_0[16][16][16][1];"};
  auto copyA =
      "_A_0[c4][c5][c6][0] = A[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3];";
  auto copyB =
      "_B_0[c4][c5][c6][0] = B[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3];";
  auto compute =
      "_C_0[c4][c5][c6][0] = (_A_0[c4][c5][c6][0] + _B_0[c4][c5][c6][0]);";
  auto copyC =
      "C[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3] = _C_0[c4][c5][c6][0];";
  auto sync = "__syncthreads()";

  auto code =
      emitCode({256, 128, 192, 224}, {16, 16, 16, 16}, {0, 0, 0, 0, 0, 0});

  // Order of copies may be arbitrary, but syncs must be inserted before and
  // after
  for (auto d : declarations) {
    EXPECT_TRUE(code.find(d) != std::string::npos);
  }
  auto posA = code.find(copyA);
  auto posB = code.find(copyB);
  auto posCompute = code.find(compute);
  auto posC = code.find(copyC);
  auto posSync1 = code.find(sync);
  auto posSync2 = code.rfind(sync, posCompute);
  auto posSync3 = code.find(sync, posCompute);
  auto posSync4 = code.rfind(sync);
  EXPECT_NE(posA, std::string::npos);
  EXPECT_NE(posB, std::string::npos);
  EXPECT_NE(posC, std::string::npos);
  EXPECT_NE(posCompute, std::string::npos);
  EXPECT_NE(posSync1, std::string::npos);
  EXPECT_NE(posSync2, std::string::npos);
  EXPECT_NE(posSync3, std::string::npos);
  EXPECT_NE(posSync4, std::string::npos);
  EXPECT_LT(posSync1, std::min(posA, posB));
  EXPECT_GT(posSync2, std::max(posA, posB));
  EXPECT_LT(posSync3, posC);
  EXPECT_GT(posSync4, posC);
}

TEST_F(Sum4D, CodeInnerBand) {
  auto declarations = {"__shared__ float32 _C_0[1][1][1][1];",
                       "__shared__ float32 _A_0[1][1][1][1];",
                       "__shared__ float32 _B_0[1][1][1][1];"};
  auto copyA =
      "_A_0[0][0][0][0] = A[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3];";
  auto copyB =
      "_B_0[0][0][0][0] = B[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3];";
  auto compute = "_C_0[0][0][0][0] = (_A_0[0][0][0][0] + _B_0[0][0][0][0]);";
  auto copyC =
      "C[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3] = _C_0[0][0][0][0];";
  auto sync = "__syncthreads()";

  auto code =
      emitCode({256, 128, 192, 224}, {16, 16, 16, 16}, {0, 0, 0, 0, 0, 0, 0});
  // Order of copies may be arbitrary, but syncs must be inserted before and
  // after
  for (auto d : declarations) {
    EXPECT_NE(std::string::npos, code.find(d));
  }
  auto posA = code.find(copyA);
  auto posB = code.find(copyB);
  auto posCompute = code.find(compute);
  auto posC = code.find(copyC);
  auto posSync1 = code.find(sync);
  auto posSync2 = code.rfind(sync, posCompute);
  auto posSync3 = code.find(sync, posCompute);
  auto posSync4 = code.rfind(sync);
  EXPECT_NE(posA, std::string::npos);
  EXPECT_NE(posB, std::string::npos);
  EXPECT_NE(posC, std::string::npos);
  EXPECT_NE(posCompute, std::string::npos);
  EXPECT_NE(posSync1, std::string::npos);
  EXPECT_NE(posSync2, std::string::npos);
  EXPECT_NE(posSync3, std::string::npos);
  EXPECT_NE(posSync4, std::string::npos);
  EXPECT_LT(posSync1, std::min(posA, posB));
  EXPECT_GT(posSync2, std::max(posA, posB));
  EXPECT_LT(posSync3, posC);
  EXPECT_GT(posSync4, posC);
}

class MapperMemoryPromotionSum2D : public MapperMemoryPromotion2DHelper {
 public:
  const string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
  C(i, j) = A(i, j) + B(i, j)
}
)TC";

  void checkPromoteAll(
      size_t problemSize1,
      size_t problemSize2,
      size_t tile1,
      size_t tile2,
      std::vector<size_t> childPos) {
    auto mscop = prepareScop(
        tc,
        std::unordered_map<std::string, size_t>{{"N", problemSize1},
                                                {"M", problemSize2}},
        {tile1, tile2});
    auto& scop = const_cast<Scop&>(mscop->scop());
    // Must force domain intersection for overapproximation to work
    scop.domain() = scop.domain().intersect_params(scop.globalParameterContext);
    auto ctx = scop.domain().get_ctx();
    auto groups = TensorReferenceGroup::accessedBySubtree(
        scop.scheduleRoot()->child(childPos), scop);
    LOG(INFO) << "Groups:\n" << groups;

    EXPECT_EQ(groups.size(), 3);

    USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
    isl::space blockSpace = isl::space(ctx, 2);
    blockSpace = blockSpace.set_dim_id(isl::dim_type::param, 0, BX)
                     .set_dim_id(isl::dim_type::param, 1, BY);
    isl::set blockZero =
        makeSpecializationSet(blockSpace, std::vector<int>{{0, 0}});

    // Must have groups for these tensors, in arbitrary order.
    unordered_set<string> names{"A", "B", "C"};
    for (const auto& g : groups) {
      auto name = g.first.get_name();
      EXPECT_EQ(names.count(name), 1);
      names.erase(name);
      const auto& tensorGroups = g.second;

      ASSERT_EQ(tensorGroups.size(), 1) << "Expected one group";
      auto oneGroup = tensorGroups[0].get();

      ASSERT_EQ(oneGroup->references.size(), 1)
          << "Expected one reference in group";
      if (name != "C") {
        EXPECT_TRUE(oneGroup->isReadOnly());
      }

      auto ref = oneGroup->references[0].get();
      ASSERT_EQ(oneGroup->approximation.size(), 2)
          << "Could not compute footprint for" << ref->scopedAccess;

      EXPECT_EQ(
          oneGroup->approximation[0].size,
          isl::val(ctx, std::min(tile1, problemSize1)));
      EXPECT_EQ(
          oneGroup->approximation[1].size,
          isl::val(ctx, std::min(tile2, problemSize2)));
      auto footprint = projectOutThreadParams(
          oneGroup->approximateFootprint().intersect_params(blockZero));
      int np = npoints(footprint);
      EXPECT_EQ(
          np, std::min(tile1, problemSize1) * std::min(tile2, problemSize2));

      auto schedule = partialSchedule(
          scop.scheduleRoot(), scop.scheduleRoot()->child(childPos));
      auto scopedAccess = oneGroup->originalAccesses().apply_domain(schedule);
      CHECK(scopedAccess.is_equal(oneGroup->scopedAccesses()))
          << "expected original accesses " << oneGroup->originalAccesses()
          << " to be equal to scoped accesses " << oneGroup->scopedAccesses()
          << " after applying the partial schedule " << schedule;
    }
    scop.promoteEverythingAt(childPos);
  }
};

class MapperMemoryPromotionRAW : public MapperMemoryPromotion2DHelper {
 public:
  const string tc = R"TC(
def fun(float(N, M) A) -> (B, C) {
  B(j, i) = A(j, i)
  C(j, i) = B(i, j)
}
)TC";

  void checkPromoteAll(
      size_t problemSize1,
      size_t problemSize2,
      size_t tile1,
      size_t tile2,
      std::vector<size_t> childPos) {
    auto mscop = prepareScop(
        tc,
        std::unordered_map<string, size_t>{{"N", problemSize1},
                                           {"M", problemSize2}},
        {tile1, tile2});
    auto& scop = const_cast<Scop&>(mscop->scop());
    // Must force domain intersection for overapproximation to work
    scop.domain() = scop.domain().intersect_params(scop.globalParameterContext);
    auto ctx = scop.domain().get_ctx();
    auto groups = TensorReferenceGroup::accessedBySubtree(
        scop.scheduleRoot()->child(childPos), scop);
    LOG(INFO) << "Groups:\n" << groups;

    ASSERT_EQ(groups.size(), 3);
    auto idA = isl::id(ctx, std::string("A"));
    auto idB = isl::id(ctx, std::string("B"));
    auto idC = isl::id(ctx, std::string("C"));
    EXPECT_EQ(groups.count(idA), 1) << "could not group A references";
    EXPECT_EQ(groups.count(idB), 1) << "could not group B references";
    EXPECT_EQ(groups.count(idC), 1) << "could not group C references";

    const auto& groupsB = groups.at(idB);
    ASSERT_EQ(groupsB.size(), 1) << "expected B refereces to be grouped";
    ASSERT_EQ(groupsB[0]->approximation.size(), 2) << "B should be a 2D array";
    EXPECT_EQ(
        groupsB[0]->approximation[0].size,
        isl::val(ctx, std::min(tile1, problemSize1)));
    EXPECT_EQ(
        groupsB[0]->approximation[1].size,
        isl::val(ctx, std::min(tile2, problemSize2)));

    auto t = scop.scheduleRoot()->child(childPos);
    LOG(INFO) << "Create copy for group:\n" << groupsB << "\t@" << *t;

    EXPECT_EQ(1, groupsB.size());
    LOG(INFO) << "Read: " << groupsB[0]->readFootprint();
    LOG(INFO) << "Write: " << groupsB[0]->writeFootprint();

    auto active = activeDomainPoints(scop.scheduleRoot(), t);
    LOG(INFO) << "Active: " << active;

    auto schedule = partialSchedule(scop.scheduleRoot(), t);
    auto scopedAccess = groupsB[0]->originalAccesses().apply_domain(schedule);
    CHECK(scopedAccess.is_equal(groupsB[0]->scopedAccesses()))
        << "expected original accesses " << groupsB[0]->originalAccesses()
        << " to be equal to scoped accesses " << groupsB[0]->scopedAccesses()
        << " after applying the partial schedule " << schedule;
  }

  std::unique_ptr<MappedScop> makeWithSharedGreedy(
      size_t problemSize1,
      size_t problemSize2,
      size_t tileSize1,
      size_t tileSize2,
      size_t depth,
      size_t maxSharedMemory) {
    auto mscop = prepareScop(
        tc, {{"N", problemSize1}, {"M", problemSize2}}, {tileSize1, tileSize2});
    promoteGreedilyAtDepth(
        *mscop,
        mscop->threadIdxxScheduleDepthState,
        depth,
        maxSharedMemory,
        false);
    return mscop;
  }
};

TEST_F(MapperMemoryPromotionSum2D, array42x256_tile32x32_firstBand) {
  checkPromoteAll(42, 256, 32, 32, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionSum2D, array42x256_tile32x64_firstBand) {
  checkPromoteAll(42, 256, 32, 64, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionSum2D, array42x40_tile64x64_firstBand) {
  checkPromoteAll(42, 40, 64, 64, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionRAW, array42x256_tile32x32_firstBand) {
  checkPromoteAll(42, 256, 32, 32, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionRAW, array42x40_tile64x64_firstBand) {
  checkPromoteAll(42, 40, 64, 64, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionRAW, fitAtOuterDepths) {
  auto mscop1 = makeWithSharedGreedy(42, 40, 64, 64, 1, 8192);
  EXPECT_EQ(mscop1->scop().promotedDecls().size(), 1)
      << "expected one reference group to be promoted";

  auto mscop2 = makeWithSharedGreedy(42, 40, 64, 64, 2, 8192);
  EXPECT_EQ(mscop2->scop().promotedDecls().size(), 1)
      << "expected one reference group to be promoted";

  // Note that due to bank conflict heuristic, we will allocate 32x33 arrays in
  // shraed memory which require 32x33x2x4=8448 bytes.
  auto mscop3 = makeWithSharedGreedy(42, 40, 32, 32, 2, 8448);
  EXPECT_EQ(mscop3->scop().promotedDecls().size(), 2)
      << "expected two reference groups to fit";

  auto mscop4 = makeWithSharedGreedy(42, 40, 32, 32, 2, 8447);
  EXPECT_EQ(mscop4->scop().promotedDecls().size(), 1)
      << "expected one reference group to be promoted";
}

TEST_F(MapperMemoryPromotionRAW, throwIfCopiesBelowThreads) {
  EXPECT_THROW(
      makeWithSharedGreedy(42, 40, 64, 64, 3, 8192),
      promotion::PromotionBelowThreadsException);

  EXPECT_THROW(
      makeWithSharedGreedy(42, 40, 64, 64, 4, 8192),
      promotion::PromotionBelowThreadsException);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
