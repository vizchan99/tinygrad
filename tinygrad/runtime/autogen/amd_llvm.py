import llvm.binding as llvm
from tinygrad.device import Compiler

class AMDLLVMCompiler(Compiler):
    def __init__(self, arch: str):
        self.initialize_llvm()
        self.arch = arch
        self.context = llvm.LLVMContext()
        self.pm = self._create_pass_manager()
        self.initialize_amd_target()
        super().__init__(f"compiler_hip_{arch}")
    
    def initialize_llvm(self):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        llvm.initialize_target('AMDGPU')

    def _initialize_amd_target(self):
        target_triple = f"amdgcn-amd-amdhsa-{self.arch}"
        target = llvm.Target.from_triple(target_triple)
        self.target_machine = target.create_target_machine(
            cpu=self.arch,
            features=f"+code-object-v3",
            opt_level=llvm.Optimization.Aggressive,
            reloc_model="pic",
            code_model="small"
        )
