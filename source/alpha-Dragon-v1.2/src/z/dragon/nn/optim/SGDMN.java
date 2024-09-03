/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim;

import java.util.Collection;
import java.util.Map;
import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class SGDMN extends Optimizer {
    protected float momentum, dampen, nesterov;
    protected Tensor[] V;
    
    protected float L1 = 0.0f;
    protected float L2 = 0.0f;

    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float momentum, float dampen, float nestrov) {
        this.momentum = momentum;
        this.dampen = dampen;
        this.nesterov = nestrov;
        V = Tensor.zero_like(params);
        Tensor.sync(V);
    }
    //</editor-fold>
    public SGDMN(Parameter[] params, float lr, float momentum, float dampen, float nesterov) {
        super(params, lr);
        __init__(momentum, dampen, nesterov);
    }

    public SGDMN(Collection<Parameter> params, float lr, float momentum, float dampen, float nesterov) {
        super(params, lr);
        __init__(momentum, dampen, nesterov);
    }

    public SGDMN(Map<String, Parameter> param_map, float lr, float momentum, float dampen, float nesterov) {
        super(param_map, lr);
        __init__(momentum, dampen, nesterov);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public SGDMN learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] veclocity() { return V; }
    
    public float momentum() { return momentum; }
    public SGDMN momentum(float momentum) { this.momentum = momentum; return this;}
    
    public float dampen() { return dampen; }
    public SGDMN dampen(float dampen) { this.dampen = dampen; return this; }
    
    public float nestorv() { return nesterov; }
    public SGDMN nestorv(float nestrov) { this.nesterov = nestrov; return this; }
    public SGDMN nestrov(boolean flag) { nesterov = (flag? 1 : 0); return this; }
    
    public float L1() { return L1; }
    public SGDMN L1(float L1) { this.L1 = L1; return this; }
    
    public float L2() { return L2; }
    public SGDMN L2(float L2) { this.L2 = L2; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append("{ learning_rate = ").append(lr);
        sb.append(", momentum = ").append(momentum);
        sb.append(", dampen = ").append(dampen);
        sb.append(", nestrov = ").append(nesterov);
        sb.append(", [L1, L2] = [")
                .append(L1).append(", ")
                .append(L2).append("] }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: state">
    //hyper state---------------------------------------------------------------
    @Override protected void hypher_state(State dic) {}
    @Override protected void update_hypher_state(State dic, boolean partial) { }

    //param state---------------------------------------------------------------
    protected String velocity_key(String param_name) { return param_name + ".velocity"; }
    
    @Override
    protected void param_state(State dic, int index, String paramName) {
        dic.put(paramName + ".velocity", V[index]);
    }
    
    @Override
    protected void update_param_state(State dic, boolean partial, int index, String param_name) {
        String velocity_key = velocity_key(param_name);
        V[index].set(dic.get(velocity_key), partial, "fail to load: " + velocity_key);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: update">
    @Override protected void __before_update__() {}

    @Override
    protected void __update__(int index, Tensor grad, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor v =  V[index];//momentum
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
           
        if(decay) eg.sgdmn(w, v, momentum, dampen, nesterov, grad, lr, L1, L2);
        else eg.sgdmn(w, v, momentum, dampen, nesterov, grad, lr);
    }

    @Override
    protected void __update__(int index, Collection<Tensor> grads, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor v =  V[index];//momentum
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.sgdmn(w, v, momentum, dampen, nesterov, grads, lr, L1, L2);
        else eg.sgdmn(w, v, momentum, dampen, nesterov, grads, lr);
    }

    @Override
    protected void __clear__() {
        if(V != null) { Tensor.delete(V); V = null; }
    }
    //</editor-fold>
}
