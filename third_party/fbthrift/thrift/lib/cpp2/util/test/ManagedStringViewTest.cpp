/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <thrift/lib/cpp2/util/ManagedStringView.h>

#include <gtest/gtest.h>

using namespace ::testing;
using namespace apache::thrift;

TEST(ManagedStringViewTest, Basic) {
  static std::string s1 = "foo";
  static folly::StringPiece s2 = "bar";

  ManagedStringView s = ManagedStringView::from_static(s1);
  EXPECT_EQ(s.str(), "foo");
  s1 = "qux";
  EXPECT_EQ(s.str(), "qux");
  EXPECT_EQ(s.view(), "qux");

  s = ManagedStringView::from_static(s2);
  EXPECT_EQ(s.str(), "bar");

  {
    std::string s3 = "baz";
    s = ManagedStringView(s3);
  }
  EXPECT_EQ(s.str(), "baz");
  EXPECT_EQ(s.view(), "baz");

  {
    folly::StringPiece s4 = "myverylongstringthatwon'tfitinsso";
    s = ManagedStringView(s4);
  }
  EXPECT_EQ(std::move(s).str(), "myverylongstringthatwon'tfitinsso");
  EXPECT_EQ(std::move(s).str(), "");

  {
    const char* s5 = "qux";
    s = ManagedStringView(s5);
  }
  EXPECT_EQ(s.str(), "qux");
}
