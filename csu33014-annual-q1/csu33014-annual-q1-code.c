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
  // replace the following code with vectorized code
  float sum_a = 0.0;
  float sum_b = 0.0;
  
  for ( int i = 0; i < size; i++ ) {
    sum_a = sum_a + a[i];
    sum_b = sum_b + b[i];
  }
  return sum_a * sum_b;
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
  for ( int i = 0; i < 2048; i = i+2  ) {
    a[i] = b[i]*c[i+1] + b[i+1]*c[i];
    a[i+1] = b[i]*c[i] - b[i+1]*c[i+1];
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
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] != b[i] )
      return 0;
  }
  return 1;
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
  for ( int i = 1; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}



