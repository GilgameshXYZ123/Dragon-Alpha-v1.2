/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.tensor;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Crop extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 1L;
    
    protected int[] start;
    protected int[] out_dim;
    
    public Crop(boolean inplace,int[] start, int[] out_dim) {
        super(inplace);
        this.start = start;
        this.out_dim = out_dim;
    }
     
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int[] start() { return start; }
    public final Crop start(int... start_point) {  this.start = start_point; return this; }
    
    public final int[] out_dim() { return out_dim; }
    public final Crop out_dim(int... out_dim) { this.out_dim = out_dim; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", start_point = ["); Vector.append(sb, start); sb.append("]");
        sb.append(", out_dim = ["); Vector.append(sb, out_dim); sb.append("] }");
    }
    
    @Override
    protected InlineCrop create_unit_core() {
        return new InlineCrop(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineCrop">
    public static class InlineCrop extends SimpleInplaceInline<Crop> 
    {
        transient protected int[] in_dim;
        
        public InlineCrop(Crop unit) { super(unit); }
        
        public final int[] start() { return ut.start; }
        public final int[] out_dim() { return ut.out_dim; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            in_dim = X.dim();
            return eg.crop(inplace, X, ut.start, ut.out_dim);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads?
                eg.expand(grad_inplace, deltaY, ut.start, in_dim) : 
                null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
