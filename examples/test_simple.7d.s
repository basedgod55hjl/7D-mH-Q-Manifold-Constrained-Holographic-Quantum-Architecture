section .data
x: dq 0

section .text
global _start

_start:
    push 10
    pop rax
    mov [x], rax
    mov rax, [x]
    push rax
    ret

