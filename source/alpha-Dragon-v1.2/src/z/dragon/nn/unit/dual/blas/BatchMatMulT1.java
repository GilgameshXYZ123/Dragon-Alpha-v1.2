/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.blas;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.unit.dual.DualFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class BatchMatMulT1 extends DualFunction
{
    private static final long serialVersionUID = 1L;
    
    protected boolean likeX1;
    
    public BatchMatMulT1(boolean likeX1) { this.likeX1 = likeX1; }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public final boolean likeX1() { return likeX1; }
    public BatchMatMulT1 likeX1(boolean flag) { likeX1 = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { likeX1 = ").append(likeX1).append(" }");
    }
    
    @Override
    protected InlineBatchMatMulT1 create_unit_core() {
        return new InlineBatchMatMulT1(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineBatchMatMulT1">
    public static class InlineBatchMatMulT1 extends DualCore<BatchMatMulT1>
    {
        public InlineBatchMatMulT1(BatchMatMulT1 unit) { super(unit); }
        
        public final boolean likeX1() { return ut.likeX1; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">  
        @Override
        protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
            return eg.batchMatMulT1(ut.likeX1, X1, X2);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads, 
                boolean backward_grads1, boolean backward_grads2) 
        {
            if(!backward_grads) return null;
        
            int count = 0; Tensor deltaX1 = null, deltaX2 = null;
            //(1) deltaX1[batch, K, N] = X2[batch, K, M] * deltaY^T[batch, M, N]
            //(2) deltaX2[batch, K, M] = X1[batch, K, N] * deltaY[batch, N, M]
            if(backward_grads1) { deltaX1 = eg.batchMatMulT2(holdX2(), deltaY); count++; }
            if(backward_grads2) { deltaX2 = eg.batchMatMul(holdX1(), deltaY); count++; }
            
            if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, the deltaY is not needed
                CountGc gc = new CountGc(count, deltaY);
                if(deltaX1 != null) deltaX1.dual(()-> { gc.countDown(); });
                if(deltaX2 != null) deltaX2.dual(()-> { gc.countDown(); });
            }
            return new Tensor[]{ deltaX1, deltaX2 };
        }
        //</editor-fold>
    }
    //</editor-fold>
}
