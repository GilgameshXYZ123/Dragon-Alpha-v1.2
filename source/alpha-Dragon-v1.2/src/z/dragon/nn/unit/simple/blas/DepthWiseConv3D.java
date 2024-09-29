/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.blas;

import java.util.Arrays;
import z.dragon.common.state.State;
import z.dragon.engine.Counter;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 */
public class DepthwiseConv3D extends SimpleUnit {
    private static final long serialVersionUID = 962781240122221L;
    
    protected int OC;
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    protected int OH, OW;
    protected boolean biased;
    
    transient protected Parameter W;
    transient protected Parameter B;
    
    public DepthwiseConv3D(boolean biased, 
            int out_channels,//out_channesl = in_channels * multiplier
            int kernel_height, int kernel_width,
            int stride_height, int stride_width,
            int padding_height, int padding_width,
            int output_height, int output_width)
    {
        this.biased = biased;
        this.OC = out_channels;
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = stride_height;  this.sw = stride_width;
        this.ph = padding_height; this.pw = padding_width;
        this.OH = output_height;  this.OW = output_width;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean biased() { return biased; }
    public int out_channels() { return OC; }
    
    public int[] kernel()  { return new int[]{ FH, FW }; }
    public int[] stride()  { return new int[]{ sh, sw }; }
    public int[] padding() { return new int[]{ ph, pw }; }
    public int[] fans() { return new int[] { FH*FW, FH*FW }; }//[fan_in, fan_out], out_channels = 1
    
    public int[] out_size() { return new int[] { OH, OW }; }
    public DepthwiseConv3D out_size(int... out_size) {
        if(out_size == null || out_size.length != 2) throw new IllegalArgumentException(
                "out_size == null || out_size.length != 2");
        OH = out_size[0]; OW = out_size[1];
        return this;
    }
    
    public Parameter weight_param() { return W; }
    public Tensor weight() { return W.ts(); }
    public DepthwiseConv3D weight(Tensor weight) {
        if(Tensor.isNull(weight)) throw new NullPointerException("weight is null");
        if(!weight.dimEquals(FH, FW, OC)) throw new IllegalArgumentException(String.format(
                "%s : weight.dim { got %s } != [FH, FW, OC] = [%d, %d, %d]", 
                name, Arrays.toString(weight.dim()), FH, FW, OC));
        if(W != null) W.delete();
        W.tensor(weight);
        return this;
    }
    
    public Parameter bias_param() { return B; }
    public Tensor bias() { return B.ts(); }
    public DepthwiseConv3D bias(Tensor bias) {
        if(Tensor.isNull(bias)) throw new NullPointerException("bias is null");
        if(!bias.dimEquals(OC)) throw new IllegalArgumentException(String.format(
                "%s : bias.dim { got %s } != [OC] = [%d]",
                name, Arrays.toString(bias.dim()), OC));
        if(B != null) B.delete(); 
        B.tensor(bias);
        return this;
    }
    
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public DepthwiseConv3D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { biased = ").append(biased);
        sb.append(", out_channels = ").append(OC);
        sb.append(", kernel = [").append(FH).append(", ").append(FW).append(']');
        sb.append(", stride = [").append(sh).append(", ").append(sw).append(']');
        sb.append(", padding = [").append(ph).append(", ").append(pw).append(" ]");
        sb.append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    transient private boolean baseW = true;
    transient private boolean baseB = true;
    protected final synchronized void reset_backward() { baseW = baseB = true;  }
    protected final synchronized boolean baseW() { boolean old = baseW; baseW = false; return old; }
    protected final synchronized boolean baseB() { boolean old = baseB; baseB = false; return old; }
    
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(W, B);
        
        Tensor tW = eg.empty(FH, FW, OC);//init convolutional kernel W
        W = new Parameter(tW).need_grads(true);
        eg.kaiming_uniform(tW.c(), fans(), (float)Math.sqrt(5.0));//inplace: tW
        
        if(biased) {//init bias B
            float bound = (float) (1.0 / Math.sqrt(FH * FW));
            Tensor tB = eg.uniform(-bound, bound, OC);
            B = new Parameter(tB).need_grads(true);
        }
        
        Parameter.sync(W, B);
    }
    
    @Override
    protected InlineDepthWiseConv3D create_unit_core() {
        return new InlineDepthWiseConv3D(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: param & state">
    protected String weight_key() { return name + ".weight"; }
    protected String bias_key() { return name + ".bias"; }
    
    @Override 
    public void params(Parameter.ParamSet set) { 
        set.add(W); 
        if(biased) set.add(B);
    }
    
    @Override
    public void param_map(Parameter.ParamMap<String> map) {
        map.put(weight_key(), W);
        if(biased) map.put(bias_key(), B);
    }
    
    @Override
    public void state(State dic) {
        dic.put(weight_key(), W.ts());
        if(biased) dic.put(bias_key(), B.ts());
    }
    
    @Override
    public void update_state(State dic, boolean partial) {
        W.ts().set(dic.get(weight_key()), partial, name + ": fail to update state for weight");
        if(biased) B.ts().set(dic.get(bias_key()), partial, name + ": fail to update state for bias");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineDepthWiseConv3D"> 
    public static class InlineDepthWiseConv3D extends SimpleCore<DepthwiseConv3D> {
        transient private int IH, IW, IC;
        transient private Tensor y;
         
        public InlineDepthWiseConv3D(DepthwiseConv3D unit) { super(unit); }
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public boolean biased() { return ut.biased; }
        public int out_channels() { return ut.OC; }
    
        public int[] kernel()  { return new int[]{ ut.FH, ut.FW }; }
        public int[] stride()  { return new int[]{ ut.sh, ut.sw }; }
        public int[] padding() { return new int[]{ ut.ph, ut.pw }; }
        public int[] out_size() { return new int[] { ut.OH, ut.OW }; }
        
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected void __before_forward__(Engine eg, Tensor X) { 
            ut.reset_backward();
            IH = X.dim(-3); IW = X.dim(-2); IC = X.dim(-1);
            y = (ut.pre_alloc_forward ? //X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
                    eg.alloc.depthwise_conv3D(X, ut.W.ts(), ut.OH, ut.OW, ut.sh, ut.sw, ut.ph, ut.pw) : 
                    null);
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(ut.pre_alloc_forward) { 
                Tensor out = y.c(); y = null;
                return (ut.biased ? //X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
                        eg.depthwise_conv3D_biased(out, X, ut.W.ts(), ut.sh, ut.sw, ut.ph, ut.pw, ut.B.ts()) :
                        eg.depthwise_conv3D(out, X, ut.W.ts(), ut.sh, ut.sw, ut.ph, ut.pw));
            }
            
            return (ut.biased ? //X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
                    eg.depthwise_conv3D_biased(X, ut.W.ts(), ut.OH, ut.OW, ut.sh, ut.sw, ut.ph, ut.pw, ut.B.ts()) :
                    eg.depthwise_conv3D(X, ut.W.ts(), ut.OH, ut.OW, ut.sh, ut.sw, ut.ph, ut.pw));
        }
        //</editor-fold>

        //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) 
        {
            Tensor deltaW = null, deltaB = null, deltaX = null;
            int gc_count = 0;
            
            if (ut.W.need_grads()) {//find the gradient of filters
                deltaW = eg.depthwise_conv3D_deltaW(holdX(), deltaY, ut.FH, ut.FW, ut.sh, ut.sw, ut.ph, ut.pw);
                ut.W.accumulate(ut.baseW(), deltaW);
                if (grad_inplace) gc_count++;
            }
            
            if (ut.biased && ut.B.need_grads()) {//find the gradient of bias
                deltaB = eg.field_sum(deltaY, ut.OC);
                ut.B.accumulate(ut.baseB(), deltaB);
                if(grad_inplace) gc_count++;
            }
             
            if (backward_grads) {//find the gradient of input-features
                deltaX = eg.depthwise_conv3D_deltaX(deltaY, ut.W.ts(), IH, IW, IC, ut.sh, ut.sw, ut.ph, ut.pw);
                ut.W.ts().follow(deltaX);//When compute deltaX, W can't be changed
                if(grad_inplace) gc_count++;
            }
            
            if (gc_count != 0) {//when deltaW, deltaB, deltaX are cauculated, deltaY is not needed
                Counter.CountGc gc = new Counter.CountGc(gc_count, deltaY);
                if(deltaW != null) { deltaW.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
            }
            return deltaX;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
