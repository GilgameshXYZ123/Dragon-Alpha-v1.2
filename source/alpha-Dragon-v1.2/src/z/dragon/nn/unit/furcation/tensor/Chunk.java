/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.furcation.tensor;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.furcation.FurcationCore;
import z.dragon.nn.unit.furcation.FurcateFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Chunk extends FurcateFunction
{
    protected int dimIdx;
    protected int num;
    
    public Chunk(int dimIdx, int num) {
        this.dimIdx = dimIdx;
        this.num = num;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int num() { return num; }
    public Chunk num(int num) { this.num = num; return this; }
    
    public final int dimIdx() { return dimIdx; }
    public Chunk dimIdx(int dimIdx) { this.dimIdx = dimIdx; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { dimIdx = ").append(dimIdx);
        sb.append(", num = ").append(num).append(" }");
    }
    
    @Override
    protected InlineChunk create_unit_core() { 
        return new InlineChunk(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineChunk"> 
    public static class InlineChunk extends FurcationCore<Chunk>
    {
        public InlineChunk(Chunk unit) { super(unit); }
        
        public final int num() { return ut.num; }
        public final int dimIdx() { return ut.dimIdx; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor[] __forward__(Engine eg, Tensor X) {
            return eg.chunk(X, ut.dimIdx, ut.num);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor[] deltaY, 
                boolean grad_inplace, boolean[] grad_inplaces, 
                boolean backward_grads) 
        {
            if(!backward_grads) return null;
        
            Tensor deltaX = eg.concat(ut.dimIdx, deltaY);
        
            if(grad_inplace) deltaX.dual(()-> { 
                for(int i=0; i<grad_inplaces.length; i++)
                    if(grad_inplaces[i]) deltaY[i].delete();
            });
        
            return deltaX;
        }
        //</editor-fold> 
    }
    //</editor-fold> 
}
