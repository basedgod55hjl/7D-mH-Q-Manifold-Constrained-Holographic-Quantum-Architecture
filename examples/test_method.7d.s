section .data
y: dq 0
x: dq 0

section .text
global _start

_start:
    push 10
    pop rax
    mov [x], rax
    mov rax, [x]
    push rax
    pop rax
    mov [y], rax
    push 0
    ret

