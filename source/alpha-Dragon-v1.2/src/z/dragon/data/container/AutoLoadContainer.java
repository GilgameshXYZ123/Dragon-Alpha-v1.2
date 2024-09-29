/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data.container;

import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import z.dragon.data.Pair;
import z.util.math.vector.Vector;

/**
 * 
 * @author Gilgamesh
 * @param <K>
 * @param <V> 
 */
@SuppressWarnings(value = "unchecked")
public class AutoLoadContainer<K, V> extends AbstractContainer<K, V> {
    public static interface Loader<K, V> { public Pair<K, V> load(); }
    
    public static interface Triger { public boolean needLoad(AutoLoadContainer con, int batch); }
    
    //<editor-fold defaultstate="collapsed" desc="class: Full">
    public static final class Full implements Triger 
    {
        static final Full full = new Full();
                
        @Override
        public boolean needLoad(AutoLoadContainer con, int batch) {
            return con.size() < con.capacity();
        }
    }
    //</editor-fold>
    public static Full full() { return Full.full; }
    
    //<editor-fold defaultstate="collapsed" desc="class: Update">
    public static class Update implements Triger 
    {
        private int count = 0;
        private double threshold = 0.5;

        Update(double threshold) { this.threshold = threshold; }

        @Override
        public boolean needLoad(AutoLoadContainer con, int batch) {
            if (con.size() < con.capacity())  return true;

            //get: batchSize = +batchSize
            count += batch;//load: batchSize = -batchSize
            double percent = ((float) count) / con.capacity();
            return percent >= threshold;
        }
    };
    //</editor-fold>
    public static Update update(double threshold) { return new Update(threshold); }
    
    //<editor-fold defaultstate="collapsed" desc="parameters">
    protected final int capacity;
    protected volatile int size = 0;
    protected final K[] karr; //contain keys
    protected final V[] varr; //contain values
    
    private int thread_num;
    private Loader<K, V> loader;
    private Triger triger;
    //</editor-fold>
    public AutoLoadContainer(Class<K> input_clazz, Class<V> label_clazz, int capacity) {
        super(input_clazz, label_clazz);
        if(capacity <= 0) throw new IllegalArgumentException(String.format(
                "capacity must { got %d } >= 0", capacity));
        
        this.capacity = capacity;
        this.karr = (K[]) Array.newInstance(input_clazz, capacity);
        this.varr = (V[]) Array.newInstance(label_clazz, capacity);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override final public int size() {  return size; }
    public final int capacity() { return capacity; }

    public final K[] inputs() { return karr; }
    public final V[] labels() { return varr; }
    
    public final int thread_num() { return thread_num; }
    public synchronized AutoLoadContainer<K, V> thread_num(int thread_num) {
        if(thread_num <= 0) throw new IllegalArgumentException(String.format(
                "thread_num { got %d } must >= 0", thread_num));
        this.thread_num = thread_num;
        return this; 
    }
    
    public final Loader<K, V> loader() { return loader; }
    public synchronized AutoLoadContainer<K, V> loader(Loader<K, V> loader) {
        if(loader == null) throw new NullPointerException("DataLoader is null");
        this.loader = loader;
        return this;
    }

    public final Triger triger() { return triger; }
    public synchronized AutoLoadContainer<K, V> triger(Triger triger) {
        if(triger == null) throw new NullPointerException("LoaderTriger is null");
        this.triger = triger;
        return this;
    }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" { ");
        sb.append("size = ").append(size());
        sb.append(", thread_num = ").append(thread_num);
        sb.append(", input_class = ").append(input_class());
        sb.append(", label_class= ").append(label_class());
        sb.append("}");
    }
    
    @Override
    public Map<V, Integer> class_sample_num() {
        Map<V, Integer> total = new HashMap<>(32);//get the total number of samples of each class
        for (int i=0; i<size; i++) {
            V label = varr[i]; 
            Integer num = total.get(label); if (num == null) num = 0; 
            total.put(label, ++num); 
        }
        return total;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="autoload">
    private static final ThreadFactory daemonThreadFactory = new ThreadFactory() {
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }
    };
    private ExecutorService exec;
    protected ExecutorService createExecutorService() {
        return Executors.newFixedThreadPool(thread_num, daemonThreadFactory); 
    }
    
    protected int load_batch;
    protected volatile boolean running = false;
    protected final byte[] loader_sync = new byte[0];
    protected Thread loader_thread;
    
    private void notify_loader(int batchSize) {
        synchronized(loader_sync) {
            if(running && triger.needLoad(this, batchSize)) loader_sync.notify();
        }
    }
    
    private void next_load(int batchSize) {
        try{ 
            synchronized(loader_sync) {
                if(triger.needLoad(this, -batchSize)) loader_sync.notifyAll();
                else loader_sync.wait();
            }
        }
        catch(InterruptedException e) {}
    }
    
    protected void load(int batch) {
        K[] ks = (K[]) Array.newInstance(kclazz, batch);
        V[] vs = (V[]) Array.newInstance(vclazz, batch);
        
        Future[] futures = new Future[batch];
        for(int i=0; i<batch; i++) {
            int index = i;
            futures[i] = exec.submit(()->{
                Pair<K, V> kv = loader.load();
                ks[index] = kv.input;
                vs[index] = kv.label;
            });
        }
        
        for(Future future : futures) {
            try { future.get(); } 
            catch(InterruptedException | ExecutionException e) {}
        }
        this.add(ks, vs);
    }
    
    public void start_load(int batch)  {
        if(batch <= 0) throw new IllegalArgumentException(String.format(
                "batch { got %d } must be a postive number", batch));
        if(batch > capacity) batch = capacity;
        
        synchronized(this) {
            load_batch = batch; 
            if(running) return;//return if not the first time to load
            if(exec == null || exec.isShutdown()) exec = createExecutorService();
            
            int first_load = load_batch << 1;
            if(first_load > capacity) first_load = capacity;
            this.load(first_load);
            
            loader_thread = new Thread(()-> { while(running) { next_load(load_batch); load(load_batch); }});
            loader_thread.start();
            running = true;
        }
    }
    
    public synchronized void stop_load() {
        if(!running) return;//return if the load is not on going
        if(!exec.isShutdown()) exec.shutdown(); exec = null;
        
        this.notifyAll();
        if(!loader_thread.isInterrupted()) loader_thread.interrupt();
        loader_thread = null;
        running = false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators: write">
    @Override
    public AutoLoadContainer<K, V> shuffle_sort() {
        start_read();
        int[] key = Vector.random_int_vector(size);
        Pair[] val = new Pair[size]; 
        for(int i=0; i<size; i++) val[i] = new Pair(karr[i], varr[i]);
        Vector.sort(key, val);
        start_write();
        
        start_write();
        for(int i=0; i<size; i++) {
            karr[i] = (K) val[i].input;
            varr[i] = (V) val[i].label;
        }
        finish_write();
        return this;
    }
    
    protected final void __random_swap(int n) {
        int len = (n + 1) >> 1;
        for(int i=0; i<len; i++) {
            int index1 = exr.nextInt(0, size - 1);
            int index2 = exr.nextInt(0, size - 1);
            K k = karr[index1]; karr[index1] = karr[index2]; karr[index2] = k;
            V v = varr[index1]; varr[index1] = varr[index2]; varr[index2] = v;
        }
    }
    
    @Override
    public AutoLoadContainer<K, V> shuffle_swap(double percent)  {
        if(percent <= 0) throw new IllegalArgumentException(String.format(
                "percent { got %f } must > 0", percent));
        if(percent >= Integer.MAX_VALUE) throw new IllegalArgumentException(String.format(
                "percent { got %f } must < Integer.MAX { got %d }", percent, Integer.MAX_VALUE));
        double op = (int)percent, rp =  percent - op;//fractional part;
        
        start_write();
        int rsize = (int) Math.ceil(rp * size);
        for(int i=0; i<op; i++) __random_swap(size);//integeral part
        __random_swap(rsize);//fractional part;
        finish_write();
        return this;
    }
    
    @Override
    public void add(K input, V label) {
        if(input == null || label == null) return;
        
        start_write();
        int index = (size < capacity ? size++ : exr.nextInt(0, capacity - 1));
        karr[index] = input;
        varr[index] = label;
        finish_write();
    }
    
    @Override
    public void add(K[] inputs, V[] labels) {
        if(inputs == null || labels == null) return;
        if(inputs.length != labels.length) throw new IllegalArgumentException(String.format(
                "inputs.length { got %d } != labels.length { got %d }",
                inputs.length, labels.length));
        
        start_write();
        for (int i=0; i<inputs.length; i++) {
            if(inputs[i] == null || labels[i] == null) continue;
            int index = (size < capacity ? size++ : exr.nextInt(0, capacity - 1));
            karr[index] = inputs[i];
            varr[index] = labels[i];
        }
        finish_write();
    }
    
    @Override
    public void clear() {
        start_write();
        for (int i=0; i<size; i++) { karr[i] = null; varr[i] = null; }
        size = 0;
        finish_write();
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="operators: read">
    @Override
    public Pair<K, V> get() {
        start_read();
        if(size == 0) throw new NullPointerException("The Container is empty");
        int index = exr.nextInt(0, size - 1);
        K k = karr[index];
        V v = varr[index];
        finish_read();
        
        notify_loader(1);     
        return new Pair<>(k, v);
    }
    
    @Override
    public Pair<K[], V[]> get(int batch) {
        if(batch <= 0) throw new IllegalArgumentException(String.format(
                "batch { got %d } must > 0", batch));
        
        K[] ks = (K[]) Array.newInstance(kclazz, batch);
        V[] vs = (V[]) Array.newInstance(vclazz, batch);
        
        start_read();
        if(size == 0) throw new NullPointerException("The Container is empty");
        for(int i=0; i<batch; i++) {
            int index = exr.nextInt(0, size - 1);
            ks[i] = karr[index];
            vs[i] = varr[index];
        }
        finish_read();
        
        notify_loader(batch);
        return new Pair<>(ks, vs);
    }
    
    @Override
    public DataContainer<K, V>[] split(int sub_size) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }
    
    @Override
    public DataContainer<K, V>[] class_split(float percent) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="batch_iterator">
    class BIter implements BatchIter<K[], V[]>
    {
        private int index = 0;
        
        @Override
        public BatchIter<K[], V[]> shuffle_swap(float percent) { 
            AutoLoadContainer.this.shuffle_swap(percent); 
            return this; 
        }

        @Override
        public BatchIter<K[], V[]> shuffle_sort() {
            AutoLoadContainer.this.shuffle_sort();
            return this;
        }

        @Override public BatchIter<K[], V[]> reset() { index = 0; return this; }
        @Override public final boolean hasNext() { return index < size; }
        
        //<editor-fold defaultstate="collapsed" desc="inner-code: next(batch)">
        final void next_inorder(K[] ks, V[] vs, int batch) {
            for(int i=0; i<batch; i++) {
                int mod_index = index % size;
                ks[i] = karr[mod_index];
                vs[i] = varr[mod_index];
                index++;
            }
        }
        
        final void next_random(K[] ks, V[] vs, int batch) {
            for(int i=0; i<batch; i++) {
                int mod_index = exr.nextInt(0, size - 1);
                ks[i] = karr[mod_index];
                vs[i] = varr[mod_index];
                index++;
            }
        }
        //</editor-fold>
        
        @Override
        public Pair<K[], V[]> next(int batch, boolean random) {
            if(batch <= 0) throw new IllegalArgumentException(String.format(
                    "batch { got %d } must > 0", batch));
            
            K[] ks = (K[]) Array.newInstance(kclazz, batch);
            V[] vs = (V[]) Array.newInstance(vclazz, batch);
            
            start_read();
            if(random) next_random(ks, vs, batch);
            else next_inorder(ks, vs, batch);
            finish_read();
            
            notify_loader(batch);
            return new Pair<>(ks, vs);
        }
    }
    //</editor-fold>
    @Override public BatchIter<K[], V[]> batch_iter() {  return new BIter(); } 
}
