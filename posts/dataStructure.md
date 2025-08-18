# Machine Learning Basic Concepts <!-- omit in toc -->

*Published on 2025-04-14 in [AI](../topics/ai.html)*
- [Statistical and Foundational Techniques](#statistical-and-foundational-techniques)
  - [Independent and Dependent Variables](#independent-and-dependent-variables)
  - [Linear Regression](#linear-regression)
  - [Logistic Regression](#logistic-regression)
  - [Training and Testing Set](#training-and-testing-set)
  - [Underfitting and Overfitting](#underfitting-and-overfitting)
  - [Regularization](#regularization)
  - [Imbalanced Dataset](#imbalanced-dataset)
- [Supervised, Unsupervised, and Reinforecement Learning](#supervised-unsupervised-and-reinforecement-learning)
  - [Labeled Data](#labeled-data)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Semisupervised Learning](#semisupervised-learning)
  - [Self-Supervised Learning](#self-supervised-learning)
  - [Reinforcement Learning](#reinforcement-learning)
- [NLP](#nlp)
  - [Encode-decode vs. Decode-only](#encode-decode-vs-decode-only)
  - [LSTM](#lstm)
  - [Transformer](#transformer)
  - [BERT](#bert)
- [LLM](#llm)
  - [Data cleaning process for training Data](#data-cleaning-process-for-training-data)
  - [KV caching](#kv-caching)
  - [Model Quantization](#model-quantization)
  - [BF16 vs. FP16](#bf16-vs-fp16)
  - [Finetuning](#finetuning)
    - [LoRA](#lora)
  - [RAG](#rag)
  - [Angentic RAG](#angentic-rag)
  - [Engineering](#engineering)
- [Recommender System Algorithms](#recommender-system-algorithms)
  - [CF](#cf)
  - [Explicit and Implicit Ratings](#explicit-and-implicit-ratings)
  - [Content-Based Recommender Systems](#content-based-recommender-systems)
  - [User-Based/Item-Based vs. Content-Based Recommender Systems](#user-baseditem-based-vs-content-based-recommender-systems)
  - [Matrix Factorization](#matrix-factorization)
- [Vision Algorithms](#vision-algorithms)
  - [CNN](#cnn)
  - [Transfer Learning](#transfer-learning)
  - [Generative Adversarial Networks](#generative-adversarial-networks)
  - [Additional Computer Vision Use Cases](#additional-computer-vision-use-cases)


# Array

# Linked List

### Impletementation

```python

class Node:
    def __init__(self, data: int):
        self.data = data
        self.next: Node | None = None


class LinkedList:
    def __init__(self):
        self.head: Node | None = None

    def append(self, data: int):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def insert_at_beginning(self, data: int):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def insert_at_position(self, position: int, data: int):
        if position < 0:
            raise IndexError("Position cannot be negative")

        new_node = Node(data)

        if position == 0:
            self.insert_at_beginning(data)
            return

        current = self.head
        index = 0

        while current and index < position - 1:
            current = current.next
            index += 1

        if not current:
            raise IndexError("Position out of bounds")

        new_node.next = current.next
        current.next = new_node

    def delete(self, key: int):
        current = self.head

        if current and current.data == key:
            self.head = current.next
            return

        prev = None
        while current and current.data != key:
            prev = current
            current = current.next

        if current:
            prev.next = current.next

    def reverse(self):
        prev = None
        current = self.head

        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

        self.head = prev

    def reverse_recursive(self):
        def _reverse_recursive(current: Node | None, prev: Node | None):
            if not current:
                return prev
            next_node = current.next
            current.next = prev
            return _reverse_recursive(next_node, current)

        self.head = _reverse_recursive(self.head, None)

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        print("Linked List:", elements)


ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.append(4)

ll.display()  # Linked List: [1, 2, 3, 4]

ll.reverse_recursive()
ll.display()  # Linked List: [4, 3, 2, 1]

```


# Doubly Linked List

# Stack

# Queue

# Tree

## Binary Tree

## Binary Search Tree

## AVL Tree


## Red-Black Tree


## B Tree

## Graph
