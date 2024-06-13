/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paper.net_present;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class AsyncAPIs 
{
    Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    
    void cpu_function1() {}
    void cpu_function2() {}
    
    public void function1() {
        Tensor X = eg.Gaussian(128, 128);
        Tensor W = eg.Uniform(128, 128);
        
        cpu_function1();
        
        Tensor A = eg.sigmoid(false, X.c());
        Tensor B = W.c().leakyRelu(false);
        Tensor C = eg.matMul(X, W);
       
        cpu_function2();
        
        Tensor Y = eg.sum(false, A.c(), B.c(), C.c());
        
        
        
        Y.delete();
    }
    
}
