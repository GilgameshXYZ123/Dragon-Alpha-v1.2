/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.unit.dual.DualFunction;

/**
 *
 * @author Gilgamesh
 */
public class Linear2Center extends DualFunction {
    private static final long serialVersionUID = 1L;
    
    protected int dim2;
    protected float alpha, beta, gamma;
    
    public Linear2Center(int dim2,
            float alpha, float beta, float gamma)
    {
        this.dim2 = dim2;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public int dim2() { return dim2; }
    public Linear2Center dim2(int dim2) { this.dim2 = dim2; return this; }
    
    public float alpha() { return alpha; }
    public Linear2Center alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Linear2Center beta(float beta) { this.beta = beta; return this; }
    
    public float gamma() { return gamma; }
    public Linear2Center gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ dim2 = ").append(dim2);
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    
    @Override
    protected DualCore<?> create_unit_core() {
        return new InlineLinear2Center(this);
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLinear2">
    public static class InlineLinear2Center extends DualCore<Linear2Center>  {
        protected transient int dim0;
        
        public InlineLinear2Center(Linear2Center unit) { super(unit); }
        
        public int dim2() { return ut.dim2; }
        public float alpha() { return ut.alpha; }
        public float beta() { return ut.beta; }
        public float gamma() { return ut.gamma; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
            dim0 =  X2.length() / ut.dim2;//X2.length = dim0 * dim2
            return eg.linear2_center(false, X1, X2, ut.dim2, ut.alpha, ut.beta, ut.gamma);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY,
                boolean grad_inplace, boolean backward_grads,
                boolean backward_grads1, boolean backward_grads2) 
        {
            if(!backward_grads) return null;
            
            int gc_count = 0; Tensor deltaX1 = null, deltaX2 = null;
            if (backward_grads1) { 
                deltaX1 = eg.linear(false, ut.alpha, deltaY, 0);
                gc_count++;//(1) deltaX1 = deltaY * alpha
            }
            if (backward_grads) {
                deltaX2 = eg.center_linear(deltaY, dim0, ut.dim2, ut.alpha, ut.beta);
                gc_count++;//(2) deltaX2 = field_sum: deltaY * beta
            }
            
            if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, deltaY is not needed
                CountGc gc = new CountGc(gc_count, deltaY);
                if(deltaX1 != null) deltaX1.dual(()-> { gc.countDown(); });
                if(deltaX2 != null) deltaX2.dual(()-> { gc.countDown(); });
            }
            return null;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
