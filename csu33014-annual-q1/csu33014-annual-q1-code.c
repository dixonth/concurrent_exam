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
  if(size >= 4) {                                          //check if large enough to vectorize
    float * sum_a_output = malloc(sizeof(float)*4);        //allocate memory for output vector
    float * sum_b_output = malloc(sizeof(float)*4);
    __m128 sum_a_vector = _mm_set1_ps(0.0f);               //initalise sum vectors with all 0s
    __m128 sum_b_vector = _mm_set1_ps(0.0f);
    __m128 a_vector, b_vector;
    int num_even = size - size%4;                          //calculate number of iterations
    for(int i = 0; i < num_even; i+=4) {
      a_vector = _mm_load_ps(&a[i]);                       //load vectors
      b_vector = _mm_load_ps(&b[i]);
      sum_a_vector = _mm_add_ps(sum_a_vector, a_vector);   //vector addition
      sum_b_vector = _mm_add_ps(sum_b_vector, b_vector);         
    }
    _mm_store_ps(sum_a_output, sum_a_vector);
    _mm_store_ps(sum_b_output, sum_b_vector);
    float sum_a = sum_a_output[0]+sum_a_output[1]+sum_a_output[2]+sum_a_output[3];  //combine vectors
    float sum_b = sum_b_output[0]+sum_b_output[1]+sum_b_output[2]+sum_b_output[3];
    for(int i = num_even; i < size; i++) {                 //deal with remainder (%4)
      sum_a += a[i];
      sum_b += b[i];
    }
    free(sum_a_output);                                    //free memory
    free(sum_b_output);
    return sum_a * sum_b;
  }
  else {                                                   //if not big enough to vectorise, use old function
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
  __m128 b_vector;
  __m128 our_num = _mm_set1_ps(1.5379f);              //initalise our_num vector with all 1.5379s
  int num_even = size - size%4;                       //calculate number of iterations
  for(int i = 0; i < num_even; i+=4) {
    b_vector = _mm_load_ps(&b[i]);
    b_vector = _mm_rcp_ps(b_vector);                  //reciprocal b_vector
    b_vector = _mm_sub_ps(our_num, b_vector);         //our_num - reciprocal
    _mm_store_ps(&a[i], b_vector);
  }
  for ( int i = 0; i < size; i++ ) {                  //deal with remainder (%4)
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
  __m128 a_vector, b_vector, a_mask, b_mask;        //initialise vectors used
  int num_even = size - size%4;                     //calculate number of iterations
  for(int i = 0; i < num_even; i+=4) {
    a_vector = _mm_load_ps(&a[i]);
    b_vector = _mm_load_ps(&b[i]);
    a_mask = _mm_cmpge_ps(a_vector, b_vector);      //'greater than' mask (0s if not, 1s if yes)
    b_mask = _mm_cmplt_ps(a_vector, b_vector);      //'less than' mask (0s if not, 1s if yes)
    a_vector = _mm_and_ps(a_vector, a_mask);        // AND with mask (goes to 0s if less than)
    b_vector = _mm_and_ps(b_vector, b_mask);        // AND with mask (goes to 0s if greater than)
    a_vector = _mm_or_ps(a_vector, b_vector);       // OR results in either a[i] or b[i] depending on previous calculations^
    _mm_store_ps(&a[i], a_vector);
  }
  for ( int i = 0; i < size; i++ ) {                //deal with remainder(%4)
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
  __m128 a_vector, b_vector, c_vector, product_1, product_2;
  for(int i = 0; i < 2048; i+=4) {
    b_vector = _mm_load_ps(&b[i]);
    c_vector = _mm_load_ps(&c[i]);
    product_1 = _mm_mul_ps(b_vector, c_vector);                                 //b[i]*c[i]
    product_1 = _mm_hsub_ps(product_1, product_1);                              //horizontal sub done on each vec
    c_vector = _mm_shuffle_ps(c_vector, c_vector, _MM_SHUFFLE(2, 3, 0, 1));     //shuffle elements of two arrays
    product_2 = _mm_mul_ps(b_vector, c_vector);                                 //b[i]*c[i+1]
    product_2 = _mm_hadd_ps(product_2, product_2);                              //horizontal add done on each vec
    a_vector = _mm_shuffle_ps(product_2, product_1, _MM_SHUFFLE(3, 2, 3, 2));   //shuffle elements of two arrays
    a_vector = _mm_shuffle_ps(a_vector, a_vector, _MM_SHUFFLE(3, 1, 2, 0));     //shuffle elements of two arrays
    _mm_store_ps(&a[i], a_vector);
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
  int num_even = size - size%16;                  //
  int num_even_floats = num_even / 4;             //calculate number of iterations (as floats instead of char)
  float * a_as_floats = (float *) b;
  float * b_as_floats = (float *) a;
  __m128 a_vector, b_vector;
  int result = 1;                                 //initialise result to true
  for(int i = 0; i < num_even_floats; i+=4) {
    a_vector = _mm_load_ps(&a_as_floats[i]);
    b_vector = _mm_load_ps(&b_as_floats[i]);
    __m128 r = _mm_cmpeq_ps(a_vector, b_vector);  //cmp a and b (all 1s if the same)
    if( _mm_movemask_ps(r) != 0xF) return 0;      //if not == (ie all 1s), return false
  }
  a = (unsigned char * restrict) a_as_floats;     //deal with remainder(%16)
  for ( int i = num_even; i < size; i++ ) {
    if ( a[i] != b[i] ) return  0;                //if not ==, return false
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
  a[0] = 0.0;                             
  for ( int i = 1; i < 4; i++ ) {                   //pass through once with old code so that b[i-1] works below
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }

  __m128 b_vector_1, b_vector_2, product_1, product_2, sum_1, sum_2, a_vector;
  __m128 c_vector_1 = _mm_setr_ps(c[0], c[1], c[2], 0.0f);    //initialise vector with 0 in c[3]
  __m128 c_vector_2 = _mm_setr_ps(0.0f, c[0], c[1], c[2]);    //initialise vector with 0 in c[0]

  for(int i = 4; i < 1020; i+=4) {
    b_vector_1 = _mm_setr_ps(b[i-1], b[i], b[i+1], b[i+2]);   //initialise vectors
    b_vector_2 = _mm_setr_ps(b[i+1], b[i+2], b[i+3], b[i+4]);
    product_1 = _mm_mul_ps(b_vector_1, c_vector_1);           //multiply vectors
    product_2 = _mm_mul_ps(b_vector_1, c_vector_2);
    sum_1 = _mm_hadd_ps(product_1, product_2);                //horizontal add done on each vector
    product_1 = _mm_mul_ps(b_vector_2, c_vector_1);           //multiply vectors
    product_2 = _mm_mul_ps(b_vector_2, c_vector_2);
    sum_2 = _mm_hadd_ps(product_1, product_2);                //horizontal add done on each vector
    a_vector = _mm_hadd_ps(sum_1, sum_2);                     //horizontal add done on each vector
    _mm_store_ps(&a[i], a_vector);
  }

  for ( int i = 1020; i < 1023; i++ ) {                       //deal with remainder (%4)
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }

  a[1023] = 0.0;
}



