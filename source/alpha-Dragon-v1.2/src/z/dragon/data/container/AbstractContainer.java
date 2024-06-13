/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data.container;

import java.util.Random;
import z.util.math.ExRandom;

/**
 *
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public abstract class AbstractContainer<K, V> implements DataContainer<K, V> 
{
    protected final ExRandom exr = new ExRandom();
    protected final Class<K> kclazz;
    protected final Class<V> vclazz;
    
    protected AbstractContainer(Class<K> input_clazz, Class<V> label_clazz) {
        if(input_clazz == null) throw new NullPointerException("input_class<K> is null");
        if(label_clazz == null) throw new NullPointerException("label_class<V> is null");
        
        this.kclazz = input_clazz;
        this.vclazz = label_clazz;
    }
   
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public Class<K> input_class() { return kclazz; }
    @Override public Class<V> label_class() { return vclazz; }
    @Override public Random random() { return exr; }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" { ");
        sb.append("size = ").append(size());
        sb.append(", input_class").append(input_class());
        sb.append(", label_class").append(label_class());
        sb.append("}");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(256);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="read & write locks">
    private volatile boolean write_lock = false;
    private volatile boolean read_lock = false;
    
    protected synchronized final void start_write() {
        try { while(write_lock || read_lock) this.wait(); } catch(InterruptedException e) {}
        read_lock = true; 
    }
    protected synchronized final void finish_write() {
        read_lock = false; 
        this.notifyAll();
    }
    
    protected synchronized final void start_read() {
        try { while(read_lock) this.wait(); } catch(InterruptedException e) {}
        write_lock = true;
    }
    protected synchronized final void finish_read() {
        write_lock = false;
        this.notifyAll();
    }
    //</editor-fold>
}
