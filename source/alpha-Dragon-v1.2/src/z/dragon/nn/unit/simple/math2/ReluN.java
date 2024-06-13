/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.nn.unit.simple.SimpleInplaceInline;

/**
 *
 * @author Gilgamesh
 */
public class ReluN extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 629971624120001L;
    
    protected float N;
    
    public ReluN(boolean inplace, float N) {
        super(inplace);
        this.N = N;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public float N() { return N; }
    public ReluN N(float N) { this.N = N; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", N = ").append(N).append(" }");
    }
    
    @Override
    protected SimpleCore<?> create_unit_core() {
        return new InlineReluN(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineReluN">
    public static class InlineReluN extends SimpleInplaceInline<ReluN> 
    {
        public InlineReluN(ReluN unit) { super(unit); }

        public float N() { return ut.N; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.reluN(inplace, X, ut.N);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdY() ?
                    eg.reluN_deltaX_v1(grad_inplace, deltaY, holdY(), ut.N) ://V1: Y is not changed
                    eg.reluN_deltaX_v2(grad_inplace, deltaY, holdX(), ut.N));//V2: X is not changed
        }
        //</editor-fold>
    }
    //</editor-fold>
}
