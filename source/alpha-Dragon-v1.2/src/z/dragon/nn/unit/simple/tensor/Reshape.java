/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.tensor;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Reshape extends SimpleInplaceFunction {
    private static final long serialVersionUID = 1L;
    
    protected int[] outDim;
    
    public Reshape(boolean inplace, int...outDim){ 
        super(inplace); 
        this.outDim = outDim;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int[] out_dim() { return outDim; }
    public final Reshape output_dim(int...outDim) { this.outDim = outDim; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", outDim = [");Vector.append(sb, outDim); sb.append("] }");
    }
    
    @Override
    protected InlineReshape create_unit_core() {
        return new InlineReshape(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineReshape">
    public static class InlineReshape extends SimpleInplaceInline<Reshape>
    {
        transient protected int[] inDim;

        public InlineReshape(Reshape unit) { super(unit); }
        
        public final int[] in_dim() { return inDim; }
        public final int[] out_dim() { return ut.outDim; }
     
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            inDim = X.dim();
            return eg.reshape(inplace, X, ut.outDim);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads?
                    eg.reshape(grad_inplace, deltaY, inDim):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
