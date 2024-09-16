from enum import Enum
from sys import argv
from io import StringIO

count = 0
registers = [0 for _ in range(7)]

def auto(reset=False) -> int:
    global count
    if reset:
        count = 0
        reset = False
    ret = count
    count += 1
    return ret

class Keyword(Enum):
    If = auto(),
    IfStar = auto(),
    Else = auto(),
    End = auto(),
    While = auto(),
    Do = auto(),
    Include = auto(),
    Memory = auto(),
    Proc = auto(),
    Const = auto(),
    Offset = auto(),
    Reset = auto(),
    Assert = auto(),
    In = auto(),
    Bikeshedder = auto(),
    Inline = auto(),
    Here = auto(),
    AddrOf = auto(),
    CallLike = auto(),
    Let = auto(),
    Peek = auto(),

class Intrinsic(Enum):
    Plus = auto(reset=True),
    Minus = auto(),
    Mult = auto(),
    Divmod = auto(),
    IDivmod = auto(),
    Max = auto(),
    Eq = auto(),
    Gt = auto(),
    Lt = auto(),
    Ge = auto(),
    Le = auto(),
    Ne = auto(),
    Shr = auto(),
    Shl = auto(),
    Or = auto(),
    And = auto(),
    Not = auto(),
    Print = auto(),
    Dup = auto(),
    Swap = auto(),
    Drop = auto(),
    Over = auto(),
    Rot = auto(),
    Load8 = auto(),
    Store8 = auto(),
    Load16 = auto(),
    Store16 = auto(),
    Load32 = auto(),
    Store32 = auto(),
    Load64 = auto(),
    Store64 = auto(),
    CastPtr = auto(),
    CastInt = auto(),
    CastBool = auto(),
    CastAddr = auto(),
    Argc = auto(),
    Argv = auto(),
    Envp = auto(),
    Syscall0 = auto(),
    Syscall1 = auto(),
    Syscall2 = auto(),
    Syscall3 = auto(),
    Syscall4 = auto(),
    Syscall5 = auto(),
    Syscall6 = auto(),
    NotIntrinsic = auto(),

class Register(Enum):
    Rax = auto(reset=True),
    Rdi = auto(),
    Rsi = auto(),
    Rdx = auto(),
    R10 = auto(),
    R8 = auto(),
    R9 = auto()

def setreg(reg, val):
    global registers
    registers[reg] = val

def getreg(reg) -> int:
    global registers
    return registers[reg]

def linux_syscall(syscalln):
    pass


def match_op(op: str) -> Intrinsic:
    match op:

        case "+":
            return Intrinsic.Plus

        case '-':
            return Intrinsic.Minus

        case 'divmod':
            return Intrinsic.Divmod

        case "idivmod":
            return Intrinsic.IDivmod

        case "*":
            return Intrinsic.Mult

        case "max":
            return Intrinsic.Max

        case "=":
            return Intrinsic.Eq

        case ">":
            return Intrinsic.Gt

        case "<":
            return Intrinsic.Lt

        case ">=":
            return Intrinsic.Ge

        case "<=":
            return Intrinsic.Le

        case "!=":
            return Intrinsic.Ne

        case "not":
            return Intrinsic.Not

        case "print":
            return Intrinsic.Print

        case "and":
            return Intrinsic.And

        case "or":
            return Intrinsic.Or

        case "shr":
            return Intrinsic.Shr

        case "shl":
            return Intrinsic.Shl

        case "dup":
            return Intrinsic.Dup

        case "swap":
            return Intrinsic.Swap

        case "over":
            return Intrinsic.Over

        case "rot":
            return Intrinsic.Rot

        case "drop":
            return Intrinsic.Drop

        case "!8":
            return Intrinsic.Store8

        case "@8":
            return Intrinsic.Load8

        case "!16":
            return Intrinsic.Store16

        case "@16":
            return Intrinsic.Load16

        case "!32":
            return Intrinsic.Store32

        case "@32":
            return Intrinsic.Load32

        case "!64":
            return Intrinsic.Store64

        case "@64":
            return Intrinsic.Load64

        case "argc":
            return Intrinsic.Argc

        case "argv":
            return Intrinsic.Argv

        case "envp":
            return Intrinsic.Envp

        case "syscall0":
            return Intrinsic.Syscall0

        case "syscall1":
            return Intrinsic.Syscall1

        case "syscall2":
            return Intrinsic.Syscall2

        case "syscall3":
            return Intrinsic.Syscall3

        case "syscall4":
            return Intrinsic.Syscall4

        case "syscall5":
            return Intrinsic.Syscall5

        case "syscall6":
            return Intrinsic.Syscall6

        case "cast(bool)":
            return Intrinsic.CastBool

        case "cast(ptr)":
            return Intrinsic.CastPtr

        case "cast(int)":
            return Intrinsic.CastInt

        case "cast(addr)":
            return Intrinsic.CastAddr

        case _:
            return Intrinsic.NotIntrinsic

class TokenType(Enum):
    Int = auto(reset=True),
    Word = auto(),
    Keyword = auto(),
    Str = auto(),
    CStr = auto(),
    Char = auto(),
    Bool = auto(),
    Ptr = auto(),
    Addr = auto(),
    Operation = auto(),

class Node:
    def __init__(self, line, pos) -> None:
        self.line = line
        self.pos = pos
        self.value = None
        self.type = None

class PostfixNode(Node):
    def __init__(self, line, pos) -> None:
        super().__init__(line, pos)

class IntNode(PostfixNode):
    def __init__(self, line, pos, num) -> None:
        super().__init__(line, pos)
        self.type = TokenType.Int
        self.value = num
    def __repr__(self) -> str:
        return f"(Int : {self.line}:{self.pos} - {self.value})"

class OpNode(PostfixNode):
    def __init__(self, line, pos, op) -> None:
        super().__init__(line, pos)
        self.type = TokenType.Operation
        self.value = op
    def __repr__(self) -> str:
        return f"(Intrinsic : {self.line}:{self.pos} - {self.value})"

class InfixNode(Node):
    pass

class WhileNode(InfixNode):
    def __init__(self, line, pos, cond, body) -> None:
        super().__init__(line, pos)
        self.cond = cond
        self.body = body
        self.type = TokenType.Keyword

class IfNode(PostfixNode):
    def __init__(self, line, pos) -> None:
        super().__init__(line, pos)
        self.type = TokenType.Keyword

class ProcNode(InfixNode):
    def __init__(self, line, pos, name: str, body: list[Node], inline: bool) -> None:
        super().__init__(line, pos)
        self.name = name
        self.body = body
        self.inline = inline
        self.type = TokenType.Keyword

class ConstNode(Node):
    def __init__(self, line, pos, sym, body) -> None:
        super().__init__(line, pos)
        self.symbol = sym
        self.body = body
        self.type = TokenType.Keyword

class MemoryNode(Node):
    def __init__(self, line, pos, sym, body) -> None:
        super().__init__(line, pos)
        self.symbol = sym
        self.body = body
        self.type = TokenType.Keyword

class CStrNode(PostfixNode):
    def __init__(self, line, pos, string) -> None:
        super().__init__(line, pos)
        self.value = string
        self.type = TokenType.CStr

class StrNode(PostfixNode):
    def __init__(self, line, pos, string) -> None:
        super().__init__(line, pos)
        self.value = string
        self.type = TokenType.Str

class FnDef:
    def __init__(self, name, args, body) -> None:
        self.name = name
        self.args = args
        self.body = body

class Env:
    def __init__(self) -> None:
        self.stack = []
        self.bindings = {}
        self.funcs = {}
    def push(self, x, type: TokenType):
        self.stack.append((x, type))
    def pop(self):
        return self.stack.pop()
    def drop(self):
        self.stack.pop()
    def swap(self):
        top = self.stack[-1]
        btop = self.before_last()
        self.stack[-1] = btop
        self.stack[len(self.stack)-2] = top
    def over(self):
        btop = self.before_last()
        self.stack.append(btop)
    def rot(self):
        pass
    def before_last(self):
        return self.stack[len(self.stack)-2]

class Parser:
    def __init__(self, syntax: str, row=None, col=None) -> None:
        self.txt = syntax
        self.program = [line.split() for line in syntax.split('\n')]
        #print(self.program)
        if row is None or col is None:
            self.row = 0
            self.col = 0
        else:
            self.row = row
            self.col = col
        self.token_pos = 0
    def is_empty(self) -> bool:
        return self.program == [[]] or self.program == []
    def line_num(self):
        return self.row + 1
    def col_pos(self):
        return self.col + 1
    def has_next(self) -> bool:
        return self.row < len(self.program) or (self.row == len(self.program)-1 and self.col < len(self.program[self.row]))
    def has_next_line(self) -> bool:
        return self.row < len(self.program)-1
    def line_has_next(self) -> bool:
        return self.col < len(self.program[self.row]) - 1
    def next_line(self):
        if self.has_next_line():
            self.row += 1
            self.col = 0
    def next(self) -> str:
        res = self.program[self.row][self.col]
        self.col += 1
        if self.col >= len(self.program[self.row]):
            self.col = 0
            self.row += 1
        self.token_pos = 0
        return res
    def next_chr(self) -> chr:
        res = self.program[self.row][self.col][self.token_pos]
        if self.token_pos < len(self.program[self.row][self.col]):
            self.token_pos += 1
        return res
    def current(self) -> str:
        return self.program[self.row][self.col]
    def current_line(self) -> str:
        return " ".join(self.program[self.row])
    def next_end(self) -> str:
        res = StringIO()

        expected_ends = 1

        while self.has_next():
            syntax = self.next()
            match syntax:

                case "end":
                    expected_ends -= 1
                    if expected_ends == 0:
                        res.write(syntax)
                        break

                case "if" | "memory" | "const" | "proc" | "let" | "peek" | "while":
                    expected_ends += 1

                case _:
                    pass

            res.write(syntax)
            res.write(" ")

        if not res.getvalue().endswith("end"):
            print(res.getvalue())
            print("Error: 'end' expected, found end of file instead.")
            exit(1)

        return res.getvalue()
    def next_quote(self):
        res = StringIO()

        while self.has_next():
            token = self.next()
            if (token.endswith("\"") or token.endswith("\"c")) and token[len(token)-2] != '\\':
                res.write(token)
                return res.getvalue()

            res.write(token + " ")

        print("Error: string is not closed.")
        exit(1)

def is_number(syntax: str) -> bool:
    try:
        int(syntax)
        return True
    except ValueError:
        return False

def is_string(syntax: str) -> bool:
    return syntax.startswith("\"") and syntax.endswith("\"")

def is_cstr(syntax: str) -> bool:
    return syntax.startswith("\"") and syntax.endswith("\"c")

def parse_string(string: str, is_cstr=False) -> str:
    res = StringIO()

    skip = False
    for idx, char in enumerate(string):

        if skip:
            skip = False
            continue

        match char:

            case '\"':
                continue

            case '\\':
                skip = True
                try:
                    match string[idx+1]:

                        case 'n':
                            res.write('\n')

                        case 'r':
                            res.write('\r')

                        case '\"':
                            res.write('\"')

                        case '\'':
                            res.write('\'')

                        case '\\':
                            res.write('\\')

                        case _:
                            print(f"Error: unrecognized escape sequence '\\{string[idx+1]}'")
                            exit(1)
                except IndexError:
                    print("Error: escape sequence expected.")
                    exit(1)

            case _:
                res.write(char)

    return res.getvalue()

   

def parse(parser: Parser) -> list[Node]:
    if parser.is_empty():
        return []
    
    tokens = []

    while parser.has_next():
        syntax = parser.next()

        if syntax.startswith("//"):
            parser.next_line()
            continue

        if is_number(syntax):
            tokens.append(IntNode(parser.line_num(), parser.col_pos(), int(syntax)))
            continue

        op = match_op(syntax)
        if op != Intrinsic.NotIntrinsic:
            tokens.append(OpNode(parser.line_num(), parser.col_pos(), op))
            continue

        if syntax.startswith("\""):
            string = syntax + " " + parser.next_quote()
            #print(string)
            if is_string(string):
                tokens.append(StrNode(parser.line_num(), parser.col_pos(), parse_string(string)))

    return tokens

def to_bytes(num, sizebits):
    shifts = sizebits//8
    bytes = []
    for i in range(shifts):
        bytes.append((num >> i*8) & 0xFF)
    return bytes

def string_to_bytes(string) -> list[int]:
    return [ord(char) for char in string]

def from_bytes(bytes):
    num = 0
    bytes.reverse()
    for byte in bytes:
        num = (num << 8) | byte
    return num

class Memory:
    def __init__(self) -> None:
        self.mem = []
    def alloc(self, size: int) -> int:
        ptr = len(self.mem)
        for _ in range(size):
            self.mem.append(0)
        return ptr
    def grab(self, numbytes, addr) -> list[int]:
        res = []
        for i in range(numbytes):
            res.append(self.mem[addr+i])
        return res
    def storebyte(self, addr, byte):
        self.mem[addr] = byte
    def storen(self, numbits, addr, item):
        bytes = to_bytes(item, numbits)
        for offset, byte in enumerate(bytes):
            try:
                self.mem[addr + offset] = byte
            except IndexError:
                print("Error: Segmentation Fault")
                exit(1)
    def loadbyte(self, addr):
        return self.mem[addr]
    def loadn(self, numbits, addr):
        return from_bytes(self.grab(numbits//8, addr))
    def store_str(self, string, ptr):
        bytes = string_to_bytes(parse_string(string))
        for idx, byte in enumerate(bytes):
            self.mem[ptr+idx] = byte

def get_innermost(string, value) -> int:
    return 0

def get_outermost(string, value) -> int:
    return 0

def get_between_indices(string, start: int, end: int) -> str:
    return ""

def get_string_between(string, start, end) -> str:
    """
    ....
    """
    if string == "":
        return ""

    result = StringIO()

    tokens = string.split()

    front = 0
    back = len(tokens) - 1

    start_found = False
    end_found = False
    while back > front:
        if start_found and end_found:
            break
        if tokens[front] == start:
            start_found = True
        elif not start_found:
            front += 1
        if tokens[back] == end:
            end_found = True
        elif not end_found:
            back -= 1

    if back < front:
        pass

    for idx in range(front+1, back):
        result.write(tokens[idx])
        if idx + 1 != back:
            result.write(" ")

    return result.getvalue()


def do_op(op: Intrinsic, env: Env, mem: Memory):
    match op:

        case Intrinsic.Plus:
            rhs = env.stack.pop()
            lhs = env.stack.pop()
            env.stack.append((lhs[0] + rhs[0], TokenType.Int))
            return

        case Intrinsic.Minus:
            rhs = env.stack.pop()
            lhs = env.stack.pop()
            env.stack.append((lhs[0] - rhs[0], TokenType.Int))
            return

        case Intrinsic.Mult:
            rhs = env.stack.pop()
            lhs = env.stack.pop()
            env.stack.append((lhs[0] * rhs[0], TokenType.Int))
            return

        case Intrinsic.Max:
            rhs = env.stack.pop()
            lhs = env.stack.pop()
            env.stack.append((max(lhs[0], rhs[0]), TokenType.Int))
            return

        case Intrinsic.Argc:
            env.stack.append((len(argv), TokenType.Int))
            return

        case Intrinsic.CastInt:
            env.stack[-1] = (env.stack[-1][0], TokenType.Int)

        case Intrinsic.CastBool:
            env.stack[-1] = (env.stack[-1][0], TokenType.Bool)

        case Intrinsic.CastPtr:
            env.stack[-1] = (env.stack[-1][0], TokenType.Ptr)

        case Intrinsic.CastAddr:
            env.stack[-1] = (env.stack[-1][0], TokenType.Addr)

        case Intrinsic.Store8:
            ptr = env.pop()
            byte = env.pop()
            mem.storebyte(ptr[0], byte[0])
        case Intrinsic.Load8:
            ptr = env.pop()
            byte = mem.loadbyte(ptr[0])
            env.push(byte, TokenType.Int)
        case Intrinsic.Store16:
            ptr = env.pop()[0]
            item = env.pop()[0]
            mem.storen(16, ptr, item)
        case Intrinsic.Load16:
            ptr = env.pop()[0]
            item = mem.loadn(16,  ptr)
            env.push(item, TokenType.Int)
        case Intrinsic.Store32:
            ptr = env.pop()[0]
            item = env.pop()[0]
            mem.storen(32, ptr, item)
        case Intrinsic.Load32:
            ptr = env.pop()[0]
            item = mem.loadn(32,  ptr)
            env.push(item, TokenType.Int)
        case Intrinsic.Store64:
            ptr = env.pop()[0]
            item = env.pop()[0]
            mem.storen(64, ptr, item)
        case Intrinsic.Load64:
            ptr = env.pop()[0]
            item = mem.loadn(64,  ptr)
            env.push(item, TokenType.Int)
        case Intrinsic.Eq:
            rhs = env.pop()
            lhs = env.pop()
            env.push(1 if rhs[0] == lhs[0] else 0, TokenType.Bool)
        case Intrinsic.Gt:
            rhs = env.pop()
            lhs = env.pop()
            env.push(1 if rhs[0] > lhs[0] else 0, TokenType.Bool)
        case Intrinsic.Lt:
            rhs = env.pop()
            lhs = env.pop()
            env.push(1 if rhs[0] < lhs[0] else 0, TokenType.Bool)
        case Intrinsic.Ge:
            rhs = env.pop()
            lhs = env.pop()
            env.push(1 if rhs[0] >= lhs[0] else 0, TokenType.Bool)
        case Intrinsic.Le:
            rhs = env.pop()
            lhs = env.pop()
            env.push(1 if rhs[0] <= lhs[0] else 0, TokenType.Bool)
        case Intrinsic.Ne:
            rhs = env.pop()
            lhs = env.pop()
            env.push(1 if rhs[0] != lhs[0] else 0, TokenType.Bool)
        case Intrinsic.Print:
            print(env.pop()[0])
        case Intrinsic.Drop:
            env.drop()
        case Intrinsic.Swap:
            env.swap()
        case _:
            print(f"{op} is not implemented.")

def interpret(abstract: list[Node], env: Env, mem: Memory):

    for token in abstract:

        if token.type == TokenType.Int:
            env.stack.append((token.value, TokenType.Int))
            continue

        if token.type == TokenType.Str:
            ptr = mem.alloc(len(token.value))
            mem.store_str(token.value, ptr)
            env.push(len(token.value), TokenType.Int)
            env.push(ptr, TokenType.Ptr)
            continue
            #print(mem.mem)

        if token.type == TokenType.Operation:
            do_op(token.value, env, mem)
            continue

def main():
    if len(argv) < 2:
        print("Error: please provide filepath.");
        exit(0)
    syntax = ""
    with open(argv[1], "r") as file:
        syntax = file.read()
    parser = Parser(syntax)
    tokens = parse(parser)
    interpret(tokens, Env(), Memory())

if __name__ == '__main__':
    main()
