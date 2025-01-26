import llvm.binding as llvm
from tinygrad.device import Compiler, CompileError
from pathlib import Path
from typing import Optional, List
import tempfile
import subprocess

class AMDLLVMCompiler(Compiler):
    def __init__(self, arch: str):
        self._initialize_llvm()
        self.arch = arch
        self.context = llvm.Context()
        self.pm = self._create_pass_manager()
        self._initialize_amd_target()
        super().__init__(f"compiler_hip_{arch}")
    
    def _initialize_llvm(self):
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
    
    def _create_pass_manager(self):
        pm = llvm.PassManager()
        pm.add('mem2reg')
        pm.add('instcombine')
        pm.add('reassociate')
        pm.add('amdgpu-promote_alloca')
        
        return pm
    
    def compile_hip(self,src:str)->bytes:
        try:
            llvm_ir = self._hip_to_llvm(src)
            module = llvm.Module(name="kernel.amd", context=self.context)
            module.triple = "amdgcn-amd-amdhsa--{self.arch}"

            module.parse_assembly(llvm_ir)
            self.pm.run(module)

            return self._generate_code_object(module)
        
        except Exception as e:
            raise CompileError(f"Failed to compile HIP code: {e}")
        
        finally:
            module.dispose()
    
    def _hip_to_llvm(self, src:str)->str:
        with tempfile.NamedTemporaryFile(suffix=".hip", mode="w") as src_file:
            src_file.write(src)
            src_file.flush()
        
        clang_cmd = ["clang++","-x","hip","-target",f"amdgcn-amd-amdhsa--{self.arch}","-emit-llvm","-S","-O3"
                     "-I/opt/rocm/include",src_file.name]
        try:
            result = subprocess.run(clang_cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise CompileError(f"Clang compilation failed: {e.stderr}")
    
    def _generate_code_object(self, module:llvm.Module)->bytes:
        module.verify()
        return self.target_machine.emit_object(module)
    
    def disassemble(self, lib:bytes):
        asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
        print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))
