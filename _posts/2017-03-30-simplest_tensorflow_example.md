---
layout: post
title:  "Simplest TensorFlow example!"
date:   2017-03-30
categories: Python, TensorFlow, Machine Learning
---

```python
import tensorflow as tf
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```


```python
num_points = 200
X = np.linspace(-2, 2, num_points)
y = X + np.random.normal(0, 1, num_points)
```


```python
plt.plot(X, y, 'r*', label='sample points')
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x10d840790>




![png](simplest%20tensorflow%20example_files/simplest%20tensorflow%20example_2_1.png)



```python
ones = np.ones(num_points)
X_matrix = np.matrix(np.column_stack((X, ones)))
y_matrix = np.transpose(np.matrix(y))
print X_matrix.shape, y_matrix.shape
```

    (200, 2) (200, 1)



```python
XtX = X_matrix.transpose().dot(X_matrix)
Xty = X_matrix.transpose().dot(y_matrix)
```


```python
solution = np.linalg.inv(XtX).dot(Xty).tolist()
```


```python
w, b = solution
```


```python
y_pred = w[0]*X + b[0]

```


```python
plt.plot(X, y, 'r*', label='sample points')
plt.plot(X, y_pred, 'b', label='predictions naive')
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x10ecc46d0>




![png](simplest%20tensorflow%20example_files/simplest%20tensorflow%20example_8_1.png)



```python
X_tensor = tf.constant(X_matrix)
y_tensor = tf.constant(y_matrix)
```


```python
XtX_tensor = tf.matmul(tf.transpose(X_tensor), X_tensor)
Xty_tensor = tf.matmul(tf.transpose(X_tensor), y_tensor)

```


```python
solution_tf = tf.matmul(tf.matrix_inverse(XtX_tensor), Xty_tensor)
```


```python
sess = tf.Session()
```


```python
soln_tf = sess.run(solution_tf).tolist()
```


```python
w_tf, b_tf = soln_tf
```


```python
y_pred_tf = w_tf[0]*X + b_tf[0]
```


```python
plt.plot(X, y, 'r*', label='sample points')
plt.plot(X, y_pred, 'b', label='predictions naive')
plt.plot(X, y_pred_tf, 'g', label='predictions naive tf')

plt.legend(loc="upper left", bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x10ede6710>




![png](simplest%20tensorflow%20example_files/simplest%20tensorflow%20example_16_1.png)



```python

```
