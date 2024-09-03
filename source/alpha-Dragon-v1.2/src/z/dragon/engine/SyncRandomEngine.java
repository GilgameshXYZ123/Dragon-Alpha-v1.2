/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
public class SyncRandomEngine {
    protected final Engine eg;
    protected SyncRandomEngine(Engine eg) { this.eg = eg; }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor uniform(Tensor X, float vmin, float vmax) {
        if(eg.check) { eg.require_dtype(X, "X"); }
        float[] value = eg.random().next_float_vector(X.length, vmin, vmax);
        return eg.set(X, value);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor gaussian(Tensor X, float mu, float sigma) {
        if(eg.check) { eg.require_dtype(X, "X"); }
        float[] value = eg.random().next_gaussianf_vector(X.length, mu, sigma);
        return eg.set(X, value);
    }
    
    //<editor-fold defaultstate="collapsed" desc="kaiming_uniform">
    public Tensor kaiming_uniform(Tensor X, int[] fans) {
        return kaiming_uniform(X, 1.0f, 
               eg.kaiming_fan_mode, fans,
               eg.kaiming_no_linearity,  null);
    }
    
    public Tensor kaiming_uniform(Tensor X, float alpha, int[] fans) {
        return kaiming_uniform(X, alpha, 
               eg.kaiming_fan_mode, fans,
               eg.kaiming_no_linearity,  null);
    }
    
    public Tensor kaiming_uniform(Tensor X, 
            int fan_mode, int[] fans,
            int nonlinearity, float... params) {
        return kaiming_uniform(X, 1.0f, 
                fan_mode, fans,
                nonlinearity, params);
    }
    
    public Tensor kaiming_uniform(Tensor X, int[] fans, float... params) {
        return kaiming_uniform(X, 1.0f, 
                eg.kaiming_fan_mode, fans,
                eg.kaiming_no_linearity, params);
    }
    
    public Tensor kaiming_uniform(Tensor X, float alpha,
            int fan_mode, int[] fans,
            int nonlinearity, float... params)
    {
        float fan = eg.fan(fan_mode, fans);
        float gain = eg.gain(nonlinearity, params);
        float std = (float) (gain / Math.sqrt(fan));
        float bound = alpha * 1.732051f * std;//sqrt(3) = 1.732051
        return this.uniform(X, -bound, bound);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="kaiming_gaussian">
    public Tensor kaiming_gaussian(Tensor X, int[] fans) {
        return kaiming_gaussian(X, 1.0f, 
                eg.kaiming_fan_mode, fans,
                eg.kaiming_no_linearity, null);
    }
    
    public Tensor kaiming_gaussian(Tensor X, float alpha, int[] fans) {
        return kaiming_gaussian(X, alpha, 
                eg.kaiming_fan_mode, fans,
                eg.kaiming_no_linearity, null);
    }
    
    public Tensor kaiming_gaussian(Tensor X, 
            int fan_mode, int[] fans, 
            int nonlinearity, float... params) {
        return kaiming_gaussian(X, 1.0f, 
                fan_mode, fans,
                nonlinearity, params);
    }
    
    public Tensor kaiming_gaussian(Tensor X, float alpha, 
            int fan_mode, int[] fans,
            int nonlinearity, float... params) 
    {
        float fan = eg.fan(fan_mode, fans);
        float gain = eg.gain(nonlinearity, params);
        float std = alpha * (float) (gain / Math.sqrt(fan));
        float sigma = alpha*std;
        return this.gaussian(X, 0, sigma);
    }
    //</editor-fold>
}
