#ifndef _IFTMATRIX_H_
#define _IFTMATRIX_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "iftCommon.h"

// Its type definition in iftBasicDataType.h
// iftMatrix

//! swig(extend = iftMatrixExt.i, destroyer = iftDestroyMatrix, name = iftMatrix)
struct ift_matrix {
    float *val;                  /* matrix's values */
    int ncols, nrows;     /* number of columns, number of rows;*/
    long *tbrow; //look-up table to speed up index acess
    long n;                        /* ncols * nrows */
    bool allocated;
};

/**
 * @brief Tensor multidimensional matrix
 * @author Peixinho
 * @date Nov, 2016
 */
typedef struct ift_tensor {
    float* val;/*Tensor values*/
    int* dimension;/*Tensor dimension*/
    int* accDimension;

    bool allocated;

    int ndimension;
    long n;/*Total number of elements*/
} iftTensor;

/**
 * @brief creates a Tensor with the dimensions specified
 * @param ndim
 * @param dim
 * @author Sep, 2017
 * @return tensor
 * @author Peixinho
 */
iftTensor* iftCreateTensorWithDimensionArray(int ndim, int* dim);

/**
 * @brief creates a Tensor
 * @param ndimension Number of dimension, followed by the specified dimensions
 * @date Nov, 2016
 * @author Peixinho
 * @return a tensor object
 */
iftTensor* iftCreateTensor(int ndimension, ...);

/**
 * @brief Destroy a tensor object
 * @author Peixinho
 * @date Nov, 2016
 * @param tensor
 */
void iftDestroyTensor(iftTensor** tensor);
/**
 * @brief Create Tensor Copy
 * @author Deangeli
 * @date may, 2017
 * @param tensor
 */
iftTensor* iftCreateTensorCopy(iftTensor* src);
iftMatrix* iftCreateMatrixPointer(float *val, int ncol, int nrow);
/**
 * @brief Reshape the matrix
 * @author Deangeli
 * @date may, 2017
 * @param matrix
 * @param newCol
 * @param newRow
 */
void iftReshapeMatrix(iftMatrix* matrix, int newCol, int newRow);
iftMatrix* iftPolynomialExpasion(iftMatrix* matrix, unsigned int degree);
void iftDestroyMatrixPointer(iftMatrix**);


/**
 * @brief Matrix of double values
 */
typedef struct ift_double_matrix {
    double *val;                  /* matrix's values */
    int ncols, nrows; // number of columns, number of rows,
    long  *tbrow; //look-up table to speed up index access
    long n;                        /* ncols * nrows */
    bool allocated;
} iftDoubleMatrix;

/**
 * @brief Matrix of integer values.
 *
 * @author Samuel Martins
 * @date October 15, 2015
 * 
 * @note Its type definition in iftBasicDataType.h
 */
// iftIntMatrix
struct ift_int_matrix {
    /** Matrix of elements. */
    int *val;
    /** Number of Columns, Rows, and look-up table to speed up index access. */
    int ncols, nrows, *tbrow;
    /** Total number of elements = ncols*nrows. */
    int n;
};


/**
 * @brief Matrix of strings of 1024 characters.
 *
 * @author Samuel Martins
 * @date October 20, 2015
 * 
 * @note Its type definition in iftBasicDataType.h
 */
// iftStrMatrix
struct ift_str_matrix {
    /** Matrix of elements. */
    char **val;
    /** Number of Columns, Rows, and look-up table to speed up index access. */
    int ncols, nrows, *tbrow;
    /** Total number of elements = ncols*nrows. */
    int n;
};

typedef struct ift_bool_matrix {
    bool *val;          // matrix's values
    int ncols, nrows;  // number of columns, number of rows,
    int  *tbrow;       //look-up table to speed up index access
    int n;             // ncols * nrows
} iftBoolMatrix;


#define iftGetMatrixCol(m,i) ((i) % (m)->ncols)
#define iftGetMatrixRow(m,i) ((i) / (m)->ncols)
#define iftGetMatrixIndex(m,c,r) ((c)+(m)->tbrow[(r)])
#define iftMatrixRowPointer(m, r) ((m)->val + iftGetMatrixIndex((m), 0, (r)))


/**
 * Macro to access the element m[c,r] in matrix m.
 * @author Peixinho
 * @date Jun, 2016
 */
#define iftMatrixElem(m, c, r) (m)->val[iftGetMatrixIndex((m), (c), (r))]


#define iftTensorElem(m, ...) (m)->val[iftGetTensorIndex((m),((m)->ndimension), __VA_ARGS__)]

void iftReshapeTensor(iftTensor *T, int ndimension, ...);

int iftGetTensorIndexDebug(iftTensor* tensor, int ndimension, ...);
int iftGetTensorIndex(iftTensor* tensor, int ndimension, ...);

//! swig(newobject)
iftMatrix *iftCreateMatrix(int ncols, int nrows);

iftDoubleMatrix* iftCopyDoubleMatrix(const iftDoubleMatrix* m);
iftDoubleMatrix *iftCreateDoubleMatrix(int ncols, int nrows);
iftMatrix *iftCopyMatrix(const iftMatrix *A);
void       iftDestroyMatrix(iftMatrix **M);
void       iftDestroyDoubleMatrix(iftDoubleMatrix **M);

//! swig(newobject)
iftMatrix *iftReadMatrix(const char *filename);

//! swig()
void        iftWriteMatrix(iftMatrix *M, char *filename);

//! swig()
void        iftPrintMatrix(const iftMatrix *M);

void        iftPrintDoubleMatrix(iftDoubleMatrix *M);
iftIntMatrix *iftReadIntMatrix(const char *filename);
void        iftWriteIntMatrix(iftIntMatrix *M, char *filename);
iftBoolMatrix *iftReadBoolMatrix(const char *filename);
void        iftWriteBoolMatrix(iftBoolMatrix *M, char *filename);

/**
 * @brief Solves the least squares problem. Finding the best x, that solves Ax=b.
 */
iftMatrix *iftLeastSquares(const iftMatrix* A, const iftMatrix* B);
iftMatrix *iftMultMatricesChain(int n, ...);
iftMatrix *iftMultMatrices(const iftMatrix *A, const iftMatrix *B);
iftMatrix *iftComputeTransformation(int n_args, ...);
void       iftMultMatrixByScalar(iftMatrix *A, double scalar );
iftMatrix *iftInvertMatrix(const iftMatrix *A);
void iftLinearCombination(float a, float* A, float b, float* B, float *C, int n);

/**
 * @brief Computes the inversion of Matrix A in place, without check for invertibility.
 */
iftMatrix * iftPseudoInvertMatrix(const iftMatrix *A);
iftDoubleMatrix *iftPseudoInvertDMatrix(const iftDoubleMatrix *A);
iftMatrix *iftTransposeMatrix(const iftMatrix *A);
double     iftMatrixDeterminant(const iftMatrix *A);
iftMatrix *iftCoFactorMatrix(const iftMatrix *A);
iftMatrix *iftRotationMatrix(char axis, float theta);
iftMatrix *iftTranslationMatrix(iftVector T);
iftMatrix *iftScaleMatrix(float sx, float sy, float sz);
iftVector  iftTransformVector(iftMatrix *M, iftVector u);
iftPoint   iftTransformPoint(iftMatrix *M, iftPoint  u);


/**
 * @brief Transform a voxel from a 4x4 transformation matrix on homogeneous coordinate.
 * 
 * @param  M 4X4 Transformation Matrix on homogeneous coordinates.
 * @param  v Voxel to be transformed 
 * @return   Transformed voxel.
 *
 * @author Samuel Martins
 * @date Apr 22, 2018
 */
iftVoxel iftTransformVoxel(const iftMatrix *M, iftVoxel v);


iftMatrix *iftIdentityMatrix(int ncols);
iftMatrix *iftRotationMatrixToAlignVectorWithZ(iftVector v);
iftMatrix *iftRotationMatrixToAlignZWithVector(iftVector v);
iftMatrix *iftExtendMatrix(const iftMatrix *M, int ncols, int nrows);
void       iftSingleValueDecomp(iftMatrix *A, iftMatrix **U, iftMatrix **S, iftMatrix **Vt);
void       iftNormalizeMatrix(iftMatrix *M);
double     iftFrobeniusNorm(iftMatrix *M);
iftMatrix  *iftReadMatrixFILE(FILE *fp);
void       iftWriteMatrixFILE(iftMatrix *M, FILE *fp);

/**
 * @brief Normalize the matrix values into the range specified by minVal and maxVal
 * @param M Matrix
 * @param minValue Minimum value
 * @param maxValue Maximum value
 * @author Cesar Castelo
 * @date Set 04, 2018
 */
void iftNormalizeMatrixInRange(iftMatrix *M, float minVal, float maxVal);

/**
 * @brief Reallocates memory for a matrix and copies the original data. The new size could be higher or lower
 * @warning If the original matrix is larger then some data will be lost, i.e. only ncols columns and nrows rows will be copied
 * @param M Matrix pointer
 * @param ncols New number of columns
 * @param nrows New number of rows
 * @author Cesar Castelo
 * @date Jul 20, 2018
 */
void iftResizeMatrix(iftMatrix **M, int ncols, int nrows);
void iftResizeDoubleMatrix(iftDoubleMatrix **M, int ncols, int nrows);
void iftResizeIntMatrix(iftIntMatrix **M, int ncols, int nrows);
void iftResizeStrMatrix(iftStrMatrix **M, int ncols, int nrows);

/**
 * Save a iftMatrix as CSV
 * @param M
 * @param filename
 * @author Peixinho
 * @date Jun, 2017
 */
void iftWriteMatrixCSV(iftMatrix* M, const char* filename);

/**
 * @brief Creates a Matrix of Integer values.
 *
 * @author Samuel Martins
 * @date October 15, 2015
 *
 * @param ncols Number of Columns.
 * @param nrows Number of Rows.
 * @return The created Integer Matrix.
 */
iftIntMatrix *iftCreateIntMatrix(int ncols, int nrows);


/**
 * @brief Destroys a Matrix of Integer values.
 *
 * @author Samuel Martins
 * @date October 15, 2015
 */
void iftDestroyIntMatrix(iftIntMatrix **iM);


/**
 * @brief Copies an integer Matrix
 * @author Samuka Martins
 * @date Sep 17, 2017
 */
iftIntMatrix *iftCopyIntMatrix(int *val, int nrows, int ncols);


/**
 * @brief Rearrange a matrix to a string for printing.
 * @author Samuka
 * @date Sep 16, 2017
 */
char *iftIntMatrixAsString(const iftIntMatrix *mat);


/**
 * @brief Rearrange a matrix to a string for printing.
 * @author Samuka
 * @date Sep 16, 2017
 */
char *iftMatrixAsString(const iftMatrix *mat);

/**
 * @brief Rearrange a matrix to a string for printing.
 * @author Samuka
 * @date Sep 16, 2017
 */
char *iftStrMatrixAsString(const iftStrMatrix *mat);


/**
 * @brief Creates a Matrix of Strings with 1024 Characters.
 *
 * @author Samuel Martins
 * @date October 20, 2015
 *
 * @param ncols Number of Columns.
 * @param nrows Number of Rows.
 * @return The created String Matrix.
 */
iftStrMatrix *iftCreateStrMatrix(int ncols, int nrows);


/**
 * @brief Destroys a Matrix of Strings.
 *
 * @author Samuel Martins
 * @date October 20, 2015
 */
void iftDestroyStrMatrix(iftStrMatrix **sM);


/**
 * @brief Copies a string matrix
 * @author Samuka Martins
 * @date Sep 17, 2017
 */
iftStrMatrix *iftCopyStrMatrix(char **val, int nrows, int ncols);

/**
 * @brief Creates a Matrix of Bool values.
 *
 * @author Cesar Castelo
 * @date Set 04, 2018
 *
 * @param ncols Number of Columns.
 * @param nrows Number of Rows.
 * @return The created Bool Matrix.
 */
iftBoolMatrix *iftCreateBoolMatrix(int ncols, int nrows);


/**
 * @brief Destroys a Matrix of Bool values.
 *
 * @author Cesar Castelo
 * @date Set 04, 2018
 */
void iftDestroyBoolMatrix(iftBoolMatrix **iM);


/**
 * @brief Copies a Bool Matrix
 * @author Cesar Castelo
 * @date Set 04, 2018
 */
iftBoolMatrix *iftCopyBoolMatrix(bool *val, int nrows, int ncols);

/**
 * @brief Transforms an array of voxels given.
 *
 * @author Thiago V. Spina
 * @date October 23, 2015

 * @param S      Transformation matrix.
 * @param points Voxel array.
 * @param n      Number of voxels in the array.
 * @return The transformed voxels.
 */
iftVoxel *iftTransformVoxels(iftMatrix *S, iftVoxel *points, int n);

/**
 * @brief Gets the minimum value among all elements in the matrix
 *
 * @author Deangeli Gomes Neves
 * @date july 4, 2016
 *
 * @param matrix input matrix.
 * @return minimum value in matrix.
 */
double iftGetMinimumValueInMatrix(iftMatrix *matrix);

/**
 * @brief Gets the maximum value among all elements in the matrix
 *
 * @author Deangeli Gomes Neves
 * @date july 4, 2016
 *
 * @param matrix input matrix.
 * @return maximumvalue in matrix.
 */
double iftGetMaximumValueInMatrix(iftMatrix *matrix);


/**
 * @brief Computes the mean value for each matrix column
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 4, 2016
 *
 * @param matrix input matrix.
 * @return vector that contains the mean value of each column
 */
iftMatrix* iftMatrixMeanColumn(const iftMatrix *matrix);

/**
 * @brief Computes the sum for each matrix column
 * @author Peixinho
 * @date Jul, 2016
 * @param matrix input matrix.
 * @return Row matrix that contains the sum of each column
 */
iftMatrix* iftMatrixSumColumn(const iftMatrix *matrix);


/**
 * @brief Computes the sum for each matrix row
 * @author Peixinho
 * @date Jul, 2016
 * @param matrix input matrix.
 * @return Column matrix that contains the sum of each row
 */
iftMatrix* iftMatrixSumRow(const iftMatrix *matrix);

/**
 * @brief Performs a mathematical operation between a matrix and a scalar
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 5, 2016
 *
 *
 *
 * @param matrix input matrix.
 * @param scalar scalar value
 * @param order 'f' to matrix[i] (operator) by scalar, 'b' to scalar (operator) by operator
 * @param operationSymbol '+' to addition, '-' to subtraction, '*' to multiplication, '/' to division.
 *
 */
void iftComputeOperationBetweenMatrixScalar(iftMatrix *matrix, double scalar, char order, char operationSymbol);

/**
 * @brief Performs a addition operation between a scalar and  all elements of matrix.
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 *
 *
 * @param matrix input matrix.
 * @param scalar scalar value
 * @param order 'f' to matrix[i][j] = matrix[i][j] +  scalar, 'b' to matrix[i][j] = scalar + matrix[i][j]
 *
 */
void iftComputeAdditionBetweenMatrixScalar(iftMatrix *matrix, double scalar, char order);

/**
 * @brief Performs a subtraction operation between a scalar and  all elements of matrix.
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * @param matrix input matrix.
 * @param scalar scalar value
 * @param order 'f' to matrix[i][j] = matrix[i][j] - scalar, 'b' to matrix[i][j] = scalar - matrix[i][j]
 *
 */
void iftComputeSubtractionBetweenMatrixScalar(iftMatrix *matrix, double scalar, char order);

/**
 * @brief Performs a multiplication operation between a scalar and  all elements of matrix.
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * @param matrix input matrix.
 * @param scalar scalar value
 * @param order 'f' to matrix[i][j] = matrix[i][j] * scalar, 'b' to matrix[i][j] = scalar * matrix[i][j]
 *
 */
void iftComputeMultiplicationBetweenMatrixScalar(iftMatrix *matrix, double scalar, char order);

/**
 * @brief Performs a division operation between a scalar and  all elements of matrix.
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * @param matrix input matrix.
 * @param scalar scalar value
 * @param order 'f' to matrix[i][j] = matrix[i][j] / scalar, 'b' to matrix[i][j] = scalar / matrix[i][j]
 *
 */
void iftComputeDivisionBetweenMatrixScalar(iftMatrix *matrix, double scalar, char order);



/**
 * @brief Performs a mathematical operation between a matrix and a vector
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 5, 2016
 *
 * if the matrix dimensons are MxN, and the input vector is a column vector, its dimensions must be
 * Mx1, otherwise 1xN.
 *
 *
 * @param matrix input matrix.
 * @param Vector row or column vector
 * @param vectorType 'r' to row vector, 'c' to column vector
 * @param operationSymbol '+' to addition, '-' to subtraction, '*' to multiplication, '/' to division.
 *
 */
void iftComputeOperationBetweenMatrixVector(iftMatrix *matrix, iftDblArray* vector, char vectorType,
                                            char operationSymbol);

void iftAdditionMatrixVectorByColumn(iftMatrix *matrix, iftMatrix* vector);
void iftAdditionMatrixVectorByRow(iftMatrix *matrix, iftMatrix* vector);
void iftSubtractionMatrixVectorByColumn(iftMatrix *matrix, iftMatrix* vector);
void iftSubtractionMatrixVectorByRow(iftMatrix *matrix, iftMatrix* vector);
void iftMultiplicationMatrixVectorByColumn(iftMatrix *matrix, iftMatrix* vector);
void iftMultiplicationMatrixVectorByRow(iftMatrix *matrix, iftMatrix* vector);
void iftDivisionMatrixVectorByColumn(iftMatrix *matrix, iftMatrix* vector);
void iftDivisionMatrixVectorByRow(iftMatrix *matrix, iftMatrix* vector);






/**
 * @brief Performs a mathematical pointwise operation between two matrices
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 5, 2016
 *
 * The matrices must have the same dimensions. The operation is A "operator" by B
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 * @param operationSymbol '+' to addition, '-' to subtraction, '*' to multiplication, '/' to division.
 *
 * @return a matrix C that is result of A (pointwise operator) by B
 */
iftMatrix* iftComputeOperationBetweenMatricesInPointWise(iftMatrix *A, iftMatrix *B, char operationSymbol);

/**
 * @brief Performs pointwise addition between two matrices
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * The matrices must have the same dimensions. The operation is A + B
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 *
 * @return a matrix C that is result of A + B
 */
iftMatrix* iftMatricesAdditionPointWise(iftMatrix *A, iftMatrix *B);

/**
 * @brief Performs pointwise subtraction between two matrices
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * The matrices must have the same dimensions. The operation is A - B
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 *
 * @return a matrix C that is result of A - B
 */
iftMatrix* iftMatricesSubtractionPointWise(iftMatrix *A, iftMatrix *B);

/**
 * @brief Performs pointwise multiplication between two matrices
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * The matrices must have the same dimensions. The operation is A * B
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 *
 * @return a matrix C that is result of A * B
 */
iftMatrix* iftMatricesMultiplicationPointWise(iftMatrix *A, iftMatrix *B);

/**
 * @brief Performs pointwise division between two matrices
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * The matrices must have the same dimensions. The operation is "A divided by B"
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 *
 * @return a matrix C that is result of "A divided by B"
 */
iftMatrix* iftMatricesDivisionPointWise(iftMatrix *A, iftMatrix *B);


/**
 * @brief Performs pointwise addition between the matrices <b>A</b> and <b>B</b>, and stores the result in <b>C</b>
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * The matrices <b>A</b> and <b>B</b> must have the same dimensions. If the matrix <b>C</b> is NULL, the function
 * allocates memory to C. If C has not the same dimensions of A/B, the functions destroy the C pointer, and then
 * creates a new <b>iftMatrix</b> pointer that matches the dimensions requirements. The operation is C = A + B
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 * @param C pointer to pointer where the result will be stored
 */
void iftMatricesAdditionPointWiseInPlace(iftMatrix *A, iftMatrix *B, iftMatrix **C);

/**
 * @brief Performs pointwise subtraction between the matrices <b>A</b> and <b>B</b>, and stores the result in <b>C</b>
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * The matrices <b>A</b> and <b>B</b> must have the same dimensions. If the matrix <b>C</b> is NULL, the function
 * allocates memory to C. If C has not the same dimensions of A/B, the functions destroy the C pointer, and then
 * creates a new <b>iftMatrix</b> pointer that matches the dimensions requirements. The operation is C = A - B
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 * @param C pointer to pointer where the result will be stored
 */
void iftMatricesSubtractionPointWiseInPlace(iftMatrix *A, iftMatrix *B, iftMatrix **C);

/**
 * @brief Performs pointwise multiplication between the matrices <b>A</b> and <b>B</b>, and stores the result in <b>C</b>
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * The matrices <b>A</b> and <b>B</b> must have the same dimensions. If the matrix <b>C</b> is NULL, the function
 * allocates memory to C. If C has not the same dimensions of A/B, the functions destroy the C pointer, and then
 * creates a new <b>iftMatrix</b> pointer that matches the dimensions requirements. The operation is C = A * B
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 * @param C pointer to pointer where the result will be stored
 */
void iftMatricesMultiplicationPointWiseInPlace(iftMatrix *A, iftMatrix *B, iftMatrix **C);

/**
 * @brief Performs pointwise division between the matrices <b>A</b> and <b>B</b>, and stores the result in <b>C</b>
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 12, 2016
 *
 * The matrices <b>A</b> and <b>B</b> must have the same dimensions. If the matrix <b>C</b> is NULL, the function
 * allocates memory to C. If C has not the same dimensions of A/B, the functions destroy the C pointer, and then
 * creates a new <b>iftMatrix</b> pointer that matches the dimensions requirements. The operation is C = A / B
 *
 *
 * @param A input matrix A.
 * @param B input matrix B
 * @param C pointer to pointer where the result will be stored
 */
void iftMatricesDivisionPointWiseInPlace(iftMatrix *A, iftMatrix *B, iftMatrix **C);



/**
 * @brief Performs a mathematical pointwise operation between two matrices <b>A</b> and <b>B</b>, and stores the result
 * in <b>C</b>
 *
 * @author Deangeli Gomes Neves
 * @date july 5, 2016
 *
 * The matrices must have the same dimensions. The operation is C = A "operator" by B
 *
 * @param A input matrix A.
 * @param B input matrix B
 * @param C the matrix where the results will be stored
 * @param operationSymbol '+' to addition, '-' to subtraction, '*' to multiplication, '/' to division.
 *
 */
void iftComputeOperationBetweenMatricesInPointWiseInPlace(iftMatrix *A, iftMatrix *B,iftMatrix *C,
                                                          char operationSymbol);



void iftComputeOperationBetweenMatricesInPointWiseInPlaceSafe(iftMatrix *A, iftMatrix *B,iftMatrix **C,
                                                              char operationSymbol);


/**
 * @brief Sum of matrix elements
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 4, 2016
 *
 *
 * @param matrix input matrix.
 *
 * @return the sum of matrix elements
 */
double  iftMatrixSum(const iftMatrix *matrix);

/**
 * @brief Sets all diagonal elements of a matrix, given a value
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 4, 2016
 *
 * The matrix must be a square matrix
 *
 * @param matrix input matrix.
 * @param diagonalValue value to be set in diagonal elements of input matrix
 *
 */
void  iftSetDiagonalValue(iftMatrix *matrix, double diagonalValue);

/**
 *
 * @param m1
 * @param m2
 * @param sum
 * @return
 */
void iftSumMatricesInPlace(const iftMatrix* m1, const iftMatrix* m2, iftMatrix* sum);

iftMatrix* iftSubtractMatrices(const iftMatrix* m1, const iftMatrix* m2);
void iftSubtractMatricesInPlace(const iftMatrix* m1, const iftMatrix* m2, iftMatrix* sub);


/**
 *
 * @param m1
 * @param m2
 * @return
 */
iftMatrix* iftSumMatrices(const iftMatrix* m1, const iftMatrix* m2);

/**
 * @brief Computes the logarithm values of all elementos of matrix
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 4, 2016
 *
 *
 * @param matrix input matrix.
 *
 */
void  iftLogarithmMatrix(iftMatrix *matrix);


/**
 * @brief Checks if the matrices are equal.
 * @author Samuka
 * @date Dec 15, 2016
 */
bool iftIsMatricesEqual(const iftMatrix *A, const iftMatrix *B);



/**
 * @brief Creates a matrix that its elements are generated randomly
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 5, 2016
 *
 * @param numberRows number of rows in matrix.
 * @param numberColumns number of columns in matrix.
 * @param mean mean value amonng the matrix elements
 * @param variance variance
 *
 * @return a matrix that contains random elements values
 */
iftMatrix *iftRandomMatrix(unsigned int numberRows, unsigned int numberColumns, double mean, double variance);

/**
 * @brief Creates a matrix that its elements are all set as <b>value</b>
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 5, 2016
 *
 * @param numberRows number of rows in matrix.
 * @param numberColumns number of columns in matrix.
 * @param value value to be set in all elements
 *
 * @return a matrix that contains a specified value in elements
 */
iftMatrix *iftCreateSetMatrix(unsigned int numberRows,unsigned int numberColumns, double value);


//TODO
void iftSetMatrix(iftMatrix* matrix, double value);
void iftSetIntMatrix(iftIntMatrix* matrix, int value);


/**
 * @brief Raises by power all elements of matrix
 * @author Deangeli Gomes Neves
 * @date july 5, 2016
 *
 * @param matrix input matrix
 * @param power Power coefficient
 */
void iftPointWiseMatrixPowInPlace(iftMatrix *matrix, double power);

/**
 * @brief Raises by power all elements of matrix
 * @author Peixinho
 * @date Jul, 2016
 *
 * @param matrix input matrix
 * @param power Power coefficient
 * @return The powered matrix
 */
iftMatrix* iftPointWiseMatrixPow(iftMatrix *matrix, double power);

/**
 * @brief Computes the sum of all elements for each row vector or column vector in the input matrix
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 6, 2016
 *
 * @param matrix input matrix
 * @param vectorType 'r' to row vector, 'c' to column vector
 * @return a vector the contains the sum of elements for each row vector or column vector
 */
iftDblArray *iftComputeSumVector(iftMatrix *matrix, char vectorType);
iftMatrix* iftComputeSumVectorBycolumn(iftMatrix* matrix);
void iftComputeSumVectorBycolumnInPlace(iftMatrix* matrix,  iftMatrix **vector);
iftMatrix* iftComputeSumVectorByRow(iftMatrix* matrix);
void iftComputeSumVectorByRowInPlace(iftMatrix* matrix, iftMatrix **vector);
void iftTransposeVectorMatrix(iftMatrix* vectorMatrix);
void iftComputeSumVectorInplace(iftMatrix *matrix,iftDblArray *array , char vectorType);
void iftComputeSumVectorInplaceSafe(iftMatrix *matrix,iftDblArray **array , char vectorType);


/**
 * @brief Computes the mean of all row vectors or column vectors in the input matrix
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 6, 2016
 *
 * @param matrix input matrix
 * @param vectorType 'r' to row vector, 'c' to column vector
 * @return a vector the contains the sum of the row vectors or column vectors
 */
iftDblArray *iftComputeMeanVector(iftMatrix *matrix, char vectorType);
void iftComputeMeanVectorInPlace(iftMatrix *matrix,iftDblArray** array,char vectorType);

/**
 * @brief Create a square matrix where the diagonal elements are the vector entries
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 6, 2016
 *
 * @param vector vector
 * @return a vector the contains the sum of the row vectors or column vectors
 */
iftMatrix *iftCreateDiagonalMatrix(double *vector, long n);

void iftCreateDiagonalMatrixInPlace(double *vector, long n,iftMatrix **matrix);

/**
 * @brief Saturate the elements of matrix. The saturation may be upper bound or lower bound
 *
 *
 * @author Deangeli Gomes Neves
 * @date july 6, 2016
 *
 * @param matrix input matrix
 * @param saturationValue limit that defines the upper or lower bound
 * @param saturationType 'u' to upper bound, 'l' to lower bound.
 * @return a vector the contains the sum of the row vectors or column vectors
 */
void iftMatrixElementsSaturation(iftMatrix *matrix, double saturationValue, char saturationType);

/**
 * @brief Sets the minimum value for a given matrix.
 * @param m Input matrix
 * @param val Minimum value
 * @author Peixinho
 * @date Dec, 2016
 */
void iftMatrixMinSaturation(iftMatrix* m, float val);

iftMatrix* iftSubMatrix(iftMatrix *matrix, int startRow, int endRow, int startColumn, int endColumn);
void iftCopyMatrixInPlace(iftMatrix *A,iftMatrix *B);
void iftCopyMatrixInPlaceSafe(iftMatrix *A,iftMatrix **B);
void iftTransposeMatrixInPlace(iftMatrix *A);
void iftMultMatricesInPlace(const iftMatrix *A, const iftMatrix *B, bool transposeA, bool transposeB, iftMatrix **C);

/**
 * @brief Merges an array of iftMatrix into a new iftMatrix (vertically)
 * @param matArray Array of matrices
 * @param n Size of the array
 * @author Cesar Castelo
 * @date Jan 24, 2018
 */
iftMatrix* iftMergeMatrixArrayVertical(iftMatrix **matArray, int n);

/**
 * @brief Merges an array of iftIntMatrix into a new iftIntMatrix (horizontally)
 * @param matArray Array of int matrices
 * @param n Size of the array
 * @author Cesar Castelo
 * @date Feb 16, 2018
 */
iftIntMatrix* iftMergeIntMatrixArrayHorizontal(iftIntMatrix **matArray, int n);


/**
 * @brief Centralize a Matrix by computing the mean value for each column (one mean per column)
 * and subtracting the values of each column for its corresponding mean.
 * @param  M Input Matrix.
 * @return   Centralized Matrix.
 */
iftMatrix *iftCentralizeMatrix(const iftMatrix *M);


/**
 * @brief Normalize a Matrix by transforming each column-vector in a unit-vector.
 * @param  M Input Matrix.
 * @return   Normalized Matrix.
 */
iftMatrix *iftUnitNormMatrix(const iftMatrix *M);

/**
 * @brief Functions to convert a matrix into a set of arrays
 * @param  M Input Matrix.
 * @return Array of arrays containing the matrix's data.
 */
iftDblArray **iftMatrixToDblArraySetByRow(iftMatrix *M);
iftFloatArray **iftMatrixToFloatArraySetByRow(iftMatrix *M);
iftDblArray **iftMatrixToDblArraySetByColumn(iftMatrix *M);
iftFloatArray **iftMatrixToFloatArraySetByColumn(iftMatrix *M);


#ifdef __cplusplus
}
#endif

#endif
