# Book Logs: Machine Learning Interviews


*Published on 2025-03-12 in [AI](../topics/ai.html)*
- [Book Logs: Machine Learning Interviews](#golang-notes)
  - [Chapter 1](#chapter1)


## Chapter 1

In the main thread (which can be thought of as a process), start a goroutine that prints "hello, golang" every 50 milliseconds. Meanwhile, the main thread also prints "hello, golang" every 50 milliseconds. After printing it 10 times, the program exits. The main thread and the goroutine must run concurrently.