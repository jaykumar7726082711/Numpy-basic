# NumPy basic 
'''NumPy is the **core scientific computing library in Python**. It enables efficient handling of:
- **Vectors** (1D arrays)  
- **Matrices** (2D arrays)  
- **Tensors** (N-dimensional arrays)  

It forms the backbone of popular libraries such as:
- **Pandas** (data analysis)  
- **SciPy** (scientific computing)  
- **scikit-learn** (machine learning)  
- **TensorFlow / PyTorch** (deep learning)  '''

Understanding NumPy is essential for working with **data, AI, and ML models**.

---

## 🔹 1. Creating Arrays

Arrays are the foundation of NumPy.  
You can create them in multiple ways:

python
import numpy as np

# From Python lists
arr = np.array([1, 2, 3])

# Predefined values
zeros = np.zeros(5)        # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))     # 2x3 matrix of ones
rand = np.random.random(4) # random values
👉 NumPy arrays are stored in contiguous memory → faster than Python lists.
👉 Arrays support any number of dimensions → ndarray.

# 2. Array Arithmetic

NumPy allows element-wise operations without loops
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(a + b)   # [11 22 33]
print(a * b)   # [10 40 90]
## Broadcasting
miles = np.array([1, 2, 3])
km = miles * 1.6   # [1.6, 3.2, 4.8]
# 🔹 3. Indexing & Slicing
Just like Python lists, but extended to multiple dimensions:
arr = np.array([10, 20, 30, 40])
print(arr[1:3])   # [20 30]

matrix = np.array([[1,2,3],[4,5,6]])
print(matrix[0,1])    # 2
print(matrix[:,1])    # column -> [2,5]
# 4. Aggregation Functions
data = np.array([1, 2, 3, 4, 5])
print(data.sum())   # 15
print(data.mean())  # 3.0
print(data.std())   # 1.414
NumPy provides fast built-in aggregation:

With matrices, use axis parameter:
matrix = np.array([[1,2],[3,4]])
print(matrix.sum(axis=0)) # [4 6] (column-wise)
print(matrix.sum(axis=1)) # [3 7] (row-wise)
# 5. Matrices Creating
M = np.array([[1,2],[3,4]])
Z = np.zeros((3,3))
O = np.ones((2,4))
## Matrix Arithmetic
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print(A + B)  # element-wise add
print(A * B)  # element-wise multiply

## Dot Product (Matrix Multiplication)
print(A.dot(B))

# 6. Transpose & Reshape

Transpose
A = np.array([[1,2,3],[4,5,6]])
print(A.T)
Reshape
arr = np.arange(6)   # [0 1 2 3 4 5]
reshaped = arr.reshape((2,3))
# 🔹 7. Higher Dimensions (ND Arrays)

NumPy supports any number of dimensions:
tensor = np.ones((2,3,4))   # 3D tensor
1D → Vector

2D → Matrix

3D → Tensor (e.g., RGB image batch)
# 🔹 8. Practical Usage in ML & Data Science

## Mathematical Formulas
pred = np.array([2.5, 0.0, 2.1])
label = np.array([3.0, -0.5, 2.0])

mse = np.mean((pred - label) ** 2)
print(mse)



9. Data Representation with NumPy

Real-world data fits naturally into NumPy arrays:

📊 Tables/Spreadsheets

Represented as 2D arrays (rows × columns).

Basis for Pandas DataFrames.


🎵 Audio & Timeseries

1D arrays of sampled values.

Example: CD-quality audio → 44,100 samples per second.



# First second of audio
audio[:44100]

🖼️ Images

Grayscale → 2D array (height × width).

RGB → 3D array (height × width × 3).


image[:10,:10]  # crop 10x10 pixels
📝 Language / Text

Convert words → IDs (vocabulary).

Replace IDs with embeddings (e.g., Word2Vec, BERT).

Stored as NumPy arrays with shape: [batch_size, seq_length, embedding_dim].

