#ifndef TEST_FIXTURES_H
#define TEST_FIXTURES_H

#include "gtest/gtest.h"

#include <rmm.h>

// Base class fixture for GDF google tests that initializes / finalizes the memory manager
struct GdfTest : public ::testing::Test
{
    static void SetUpTestCase() {
        ASSERT_EQ( RMM_SUCCESS, rmmInitialize(nullptr) );
    }

    static void TearDownTestCase() {
        ASSERT_EQ( RMM_SUCCESS, rmmFinalize() );
    }
};

#endif