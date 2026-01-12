"""
Lesson 1.3: TensorFlow Tensor Basics
====================================
Understanding tensors, shapes, and operations.
Author: Akshat
Date: 2026-01-12
"""

import tensorflow as tf
import numpy as np
import sys


def main():
    print("=" * 60)
    print("    TENSORFLOW TENSOR BASICS - LESSON 1.3")
    print("=" * 60)
    print(f"\nTensorFlow Version: {tf.__version__}")
    print(f"Python Version: {sys.version}")
    print(f"NumPy Version: {np.__version__}")
    print("=" * 60 + "\n")

    # ============================================
    # SECTION 2: TENSOR RANKS
    # ============================================
    print("\n" + "=" * 60)
    print("SECTION 2: TENSOR RANKS (DIMENSIONS)")
    print("=" * 60 + "\n")

    scalar = tf.constant(42)
    print("🔹 SCALAR (Rank 0):")
    print(f"Value: {scalar.numpy()}")
    print(f"Shape: {scalar.shape}")
    print(f"Rank: {tf.rank(scalar).numpy()}\n")

    vector = tf.constant([1, 2, 3, 4, 5])
    print("🔹 VECTOR (Rank 1):")
    print(f"Value: {vector.numpy()}")
    print(f"Shape: {vector.shape}")
    print(f"Rank: {tf.rank(vector).numpy()}")
    print(f"Total elements: {tf.size(vector).numpy()}\n")

    matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
    print("🔹 MATRIX (Rank 2):")
    print(matrix.numpy())
    print(f"Shape: {matrix.shape}")
    print(f"Rank: {tf.rank(matrix).numpy()}\n")

    tensor_3d = tf.constant([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    print("🔹 3D TENSOR (Rank 3):")
    print(tensor_3d.numpy())
    print(f"Shape: {tensor_3d.shape}")
    print(f"Rank: {tf.rank(tensor_3d).numpy()}")
    print(f"Total elements: {tf.size(tensor_3d).numpy()}\n")

    # ============================================
    # SECTION 3: BASIC OPERATIONS
    # ============================================
    print("\n" + "=" * 60)
    print("SECTION 3: BASIC TENSOR OPERATIONS")
    print("=" * 60 + "\n")

    a = tf.constant([10, 20, 30, 40])
    b = tf.constant([1, 2, 3, 4])

    print(f"A: {a.numpy()}")
    print(f"B: {b.numpy()}")
    print(f"A + B: {tf.add(a, b).numpy()}")
    print(f"A - B: {tf.subtract(a, b).numpy()}")
    print(f"A * B: {tf.multiply(a, b).numpy()}")
    print(f"A / B: {tf.divide(a, b).numpy()}")
    print(f"A²: {tf.square(a).numpy()}")
    print(f"√A: {tf.sqrt(tf.cast(a, tf.float32)).numpy()}\n")

    # ============================================
    # SECTION 4: MATRIX MULTIPLICATION
    # ============================================
    print("\n" + "=" * 60)
    print("SECTION 4: MATRIX OPERATIONS")
    print("=" * 60 + "\n")

    A = tf.constant([[1., 2.], [3., 4.]])
    B = tf.constant([[5., 6.], [7., 8.]])

    print("Matrix A:\n", A.numpy())
    print("Matrix B:\n", B.numpy())
    print("\nA @ B:\n", tf.matmul(A, B).numpy())
    print("\nA * B (element-wise):\n", tf.multiply(A, B).numpy())

    # ============================================
    # SECTION 5: RESHAPING & INDEXING
    # ============================================
    print("\n" + "=" * 60)
    print("SECTION 5: RESHAPING & INDEXING")
    print("=" * 60 + "\n")

    original = tf.range(1, 13)
    reshaped = tf.reshape(original, (3, 4))

    print("Original:", original.numpy())
    print("Reshaped (3x4):\n", reshaped.numpy())
    print("First row:", reshaped[0].numpy())
    print("First column:", reshaped[:, 0].numpy(), "\n")

    # ============================================
    # SECTION 6: IMAGE SIMULATION
    # ============================================
    print("\n" + "=" * 60)
    print("SECTION 6: IMAGE SIMULATION")
    print("=" * 60 + "\n")

    fake_image = tf.constant(
        np.random.randint(0, 256, size=(4, 4, 3)),
        dtype=tf.uint8
    )

    print("Fake Image Shape:", fake_image.shape)
    normalized = tf.cast(fake_image, tf.float32) / 255.0
    print("Normalized range:", tf.reduce_min(normalized).numpy(),
          "to", tf.reduce_max(normalized).numpy(), "\n")

    # ============================================
    # SECTION 7: RANDOM TENSORS
    # ============================================
    print("\n" + "=" * 60)
    print("SECTION 7: RANDOM TENSORS")
    print("=" * 60 + "\n")

    weights = tf.random.normal((4, 3), mean=0.0, stddev=0.1)
    print("Weights shape:", weights.shape)
    print(weights.numpy(), "\n")

    print("=" * 60)
    print("END OF LESSON 1.3")
    print("=" * 60)


if __name__ == "__main__":
    main()
