//
// CSU33014 Annual Exam, May 2021
// Question 1
//

// Please examine version each of the following routines with names
// starting 'routine_'. Where the routine can be vectorized, please
// replace the corresponding 'vectorized' version using SSE vector
// intrinsics. Where it cannot be vectorized please explain why.

// To illustrate what you need to do, routine_0 contains a
// non-vectorized piece of code, and vectorized_0 shows a
// corresponding vectorized version of the same code.

// Note that to simplify testing, I have put a copy of the original
// non-vectorized code in the vectorized version of the code for
// routines 1 to 6. This allows you to easily see what the output of
// the program looks like when the original and vectorized version of
// the code produce equivalent output.

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".


#include <immintrin.h>
#include <stdio.h>

#include "csu33014-annual-q1-code.h"

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void routine_0(float * restrict a, float * restrict b,
		    float * restrict c) {
  for (int i = 0; i < 1024; i++ ) {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void vectorized_0(float * restrict a, float * restrict b,
		    float * restrict c) {
  __m128 a4, b4, c4;
  
  for (int i = 0; i < 1024; i = i+4 ) {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float routine_1(float * restrict a, float * restrict b,
		int size) {
  float sum_a = 0.0;
  float sum_b = 0.0;
  
  for ( int i = 0; i < size; i++ ) {
    sum_a = sum_a + a[i];
    sum_b = sum_b + b[i];
  }
  return sum_a * sum_b;
}

// in the following, size can have any positive value
float vectorized_1(float * restrict a, float * restrict b,
		   int size) {
  if(size >= 4) {
    float * sum_a_output = malloc(sizeof(float)*4);       
    float * sum_b_output = malloc(sizeof(float)*4);
    __m128 sum_a_vec = _mm_set1_ps(0.0f);
    __m128 sum_b_vec = _mm_set1_ps(0.0f);
    __m128 a_vec;
    __m128 b_vec;
    int num_even = size - size%4;
    for(int i = 0; i < num_even; i+=4) {
      a_vec = _mm_load_ps(&a[i]);
      b_vec = _mm_load_ps(&b[i]);
      sum_a_vec = _mm_add_ps(sum_a_vec, a_vec);
      sum_b_vec = _mm_add_ps(sum_b_vec, b_vec);
    }
    _mm_store_ps(sum_a_output, sum_a_vec);
    _mm_store_ps(sum_b_output, sum_b_vec);
    float sum_a = sum_a_output[0]+sum_a_output[1]+sum_a_output[2]+sum_a_output[3];
    float sum_b = sum_b_output[0]+sum_b_output[1]+sum_b_output[2]+sum_b_output[3];
    for(int i = num_even; i < size; i++) {
      sum_a += a[i];
      sum_b += b[i];
    }
    free(sum_a_output);
    free(sum_b_output);
    return sum_a * sum_b;
  }
  else {
    float sum_a = 0.0;
    float sum_b = 0.0;
    for ( int i = 0; i < size; i++ ) {
      sum_a = sum_a + a[i];
      sum_b = sum_b + b[i];
    }
    return sum_a * sum_b;
  }
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void routine_2(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = 1.5379 - (1.0/b[i]);
  }
}


void vectorized_2(float * restrict a, float * restrict b, int size) {
  // replace the following code with vectorized code
  __m128 b_vec;
  __m128 our_num = _mm_set1_ps(1.5379f);
  int num_even = size - size%4;
  for(int i = 0; i < num_even; i+=4) {
    b_vec = _mm_load_ps(&b[i]);
    b_vec = _mm_rcp_ps(b_vec);
    b_vec = _mm_sub_ps(our_num, b_vec);
    _mm_store_ps(&a[i], b_vec);
  }
  for ( int i = 0; i < size; i++ ) {
    a[i] = 1.5379 - (1.0/b[i]);
  }
}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void routine_3(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] < b[i] ) {
      a[i] = b[i];
    }
  }
}


void vectorized_3(float * restrict a, float * restrict b, int size) {
  // replace the following code with vectorized code
  __m128 a_vec;
  __m128 b_vec;
  __m128 a_mask;
  __m128 b_mask;

  int num_even = size - size%4;
  for(int i = 0; i < num_even; i+=4) {
    a_vec = _mm_load_ps(&a[i]);
    b_vec = _mm_load_ps(&b[i]);
    a_mask = _mm_cmpge_ps(a_vec, b_vec);
    b_mask = _mm_cmplt_ps(a_vec, b_vec);
    a_vec = _mm_and_ps(a_vec, a_mask);
    b_vec = _mm_and_ps(b_vec, b_mask);
    a_vec = _mm_or_ps(a_vec, b_vec);
    _mm_store_ps(&a[i], a_vec);
  }
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] < b[i] ) {
      a[i] = b[i];
    }
  }
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void routine_4(float * restrict a, float * restrict b,
		 float * restrict c) {
  for ( int i = 0; i < 2048; i = i+2  ) {
    a[i] = b[i]*c[i+1] + b[i+1]*c[i];
    a[i+1] = b[i]*c[i] - b[i+1]*c[i+1];
  }
}


void vectorized_4(float * restrict a, float * restrict b,
		    float * restrict  c) {
  // replace the following code with vectorized code
  __m128 a_vec;
  __m128 b_vec;
  __m128 c_vec;
  __m128 prod_1;
  __m128 prod_2;
  for(int i = 0; i < 2048; i+=4) {
    b_vec = _mm_load_ps(&b[i]);
    c_vec = _mm_load_ps(&c[i]);
    prod_1 = _mm_mul_ps(b_vec, c_vec);
    prod_1 = _mm_hsub_ps(prod_1, prod_1);
    c_vec = _mm_shuffle_ps(c_vec, c_vec, _MM_SHUFFLE(2, 3, 0, 1));
    prod_2 = _mm_mul_ps(b_vec, c_vec);
    prod_2 = _mm_hadd_ps(prod_2, prod_2);
    a_vec = _mm_shuffle_ps(prod_2, prod_1, _MM_SHUFFLE(3, 2, 3, 2));
    a_vec = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3, 1, 2, 0));
    _mm_store_ps(&a[i], a_vec);
  }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
int routine_5(unsigned char * restrict a,
	      unsigned char * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] != b[i] )
      return 0;
  }
  return 1;
}

int vectorized_5(unsigned char * restrict a,
		 unsigned char * restrict b, int size) {
  // replace the following code with vectorized code
  int num_even = size - size%16;
  int num_even_floats = num_even / 4;
  float * a_as_floats = (float *) b;
  float * b_as_floats = (float *) a;
  __m128 a_vec;
  __m128 b_vec;
  int result = 1;
  for(int i = 0; i < num_even_floats; i+=4) {
    a_vec = _mm_load_ps(&a_as_floats[i]);
    b_vec = _mm_load_ps(&b_as_floats[i]);
    __m128 r = _mm_cmpeq_ps(a_vec, b_vec);
    if( _mm_movemask_ps(r) != 0xF) result = 0;
  }
  a = (unsigned char * restrict) a_as_floats;
  for ( int i = num_even; i < size; i++ ) {
    if ( a[i] != b[i] ) result = 0;
  }
  return result;
}

/********************* routine 6 ***********************/

void routine_6(float * restrict a, float * restrict b,
		       float * restrict c) {
  a[0] = 0.0;
  for ( int i = 1; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void vectorized_6(float * restrict a, float * restrict b,
		       float * restrict c) {
  // replace the following code with vectorized code
  a[0] = 0.0;
  for ( int i = 1; i < 4; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  
  __m128 b_1;
  __m128 b_2;
  __m128 c_1 = _mm_setr_ps(c[0], c[1], c[2], 0.0f);
  __m128 c_2 = _mm_setr_ps(0.0f, c[0], c[1], c[2]);
  __m128 prod_1;
  __m128 prod_2;
  __m128 sum_1;
  __m128 sum_2;
  __m128 a_vec;
  for(int i = 4; i < 1020; i+=4) {
    b_1 = _mm_setr_ps(b[i-1], b[i], b[i+1], b[i+2]);
    b_2 = _mm_setr_ps(b[i+1], b[i+2], b[i+3], b[i+4]);
    prod_1 = _mm_mul_ps(b_1, c_1);
    prod_2 = _mm_mul_ps(b_1, c_2);
    sum_1 = _mm_hadd_ps(prod_1, prod_2);
    prod_1 = _mm_mul_ps(b_2, c_1);
    prod_2 = _mm_mul_ps(b_2, c_2);
    sum_2 = _mm_hadd_ps(prod_1, prod_2);
    a_vec = _mm_hadd_ps(sum_1, sum_2);
    _mm_store_ps(&a[i], a_vec);
  }

  for ( int i = 1020; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }

  a[1023] = 0.0;
}



