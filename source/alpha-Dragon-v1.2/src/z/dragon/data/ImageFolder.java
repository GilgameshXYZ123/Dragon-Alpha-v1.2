/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import java.io.File;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import java.util.function.Function;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.data.container.DataContainer;

/**
 *
 * @author Gilgamesh
 */
public class ImageFolder extends DataSet<byte[], Integer> {
    protected FileFolder ff;
    protected Function<File, byte[]> pixel_transform;
    protected ExecutorService img_exec;
    protected int num_threads;
    
    public ImageFolder(FileFolder file_folder, 
            Function<File, byte[]> pixel_transform, 
            int num_threads) 
    {
        super(null);
        this.ff = file_folder;
        this.pixel_transform = pixel_transform;
        num_threads(num_threads);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override  public DataSet<byte[], Integer> container(DataContainer<byte[], Integer> conta) { throw new UnsupportedOperationException("Not supported yet."); }
    @Override public DataContainer<byte[], Integer> container() { throw new UnsupportedOperationException("Not supported yet.");  }
    
    @Override public ImageFolder set_seed(long seed) { ff.set_seed(seed); return this; }

    public FileFolder fileFolder() { return ff; }
    
    @Override public Class<Integer> label_class() { return Integer.class;  }
    @Override public Class<byte[]> input_class() { return byte[].class; }
    @Override public int size() { return ff.size(); }
    
    public File root_dir() { return ff.root_dir(); }
    public ImageFolder root_dir(String root_dir) { ff.root_dir(root_dir); return this; }
    public ImageFolder root_dir(File root_dir) { ff.root_dir(root_dir); return this; }
    
    public Integer label(String dir_name) { return ff.label(dir_name); }
    public Integer[] label(String... dir_names) { return ff.label(dir_names); }
    
    public Map<String, Integer> labels() { return ff.labels(); }
    public ImageFolder labels(Map<String, Integer> labels) { ff.labels(labels); return this; }
    
    public int num_threads() { return num_threads; }
    public ImageFolder num_threads(int num_threads) {
        if(num_threads < 0) num_threads = Math.min(Runtime.getRuntime().availableProcessors(), 16);
        this.num_threads = num_threads;
        return this;
    }
    
    public Function<File, byte[]> pixel_transform() { return pixel_transform; }
    public ImageFolder pixel_transform(Function<File, byte[]> pixel_transform) {
        this.pixel_transform = pixel_transform;
        return this;
    }
    
    @Override 
    public ImageFolder input_transform(Transform<byte[][]> input_transform) { 
        super.input_transform(input_transform); return this; 
    }
    @Override 
    public ImageFolder label_transform(Transform<Integer[]> label_transform) { 
        super.label_transform(label_transform); return this; 
    }
     
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" { ");
        sb.append("size = ").append(size());
        sb.append(", input_class").append(input_class());
        sb.append(". label_class").append(label_class());
        sb.append(" inited = ").append(ff.inited);
        sb.append(", num_class = ").append(ff.labels == null ? 0 : ff.labels.size());
        sb.append(", root_dir = ").append(ff.root_dir);
        sb.append(", num_threads = ").append(num_threads);
        sb.append("}");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    //<editor-fold defaultstate="collapsed" desc="operators: inner-code">
    protected byte[][] to_pixels(File[] files) { 
        Future[] fts = new Future[files.length];
        byte[][] pixels = new byte[files.length][];
        
        for(int i=0; i<files.length; i++) {
            File file = files[i];
            fts[i] = img_exec.submit(()->{ return pixel_transform.apply(file); });
        }
        
        try { for(int i=0; i<pixels.length; i++) pixels[i] = (byte[]) fts[i].get(); }
        catch(InterruptedException | ExecutionException e) { 
            throw new RuntimeException(e); 
        }
        return pixels;
    }
    
    @Override
    protected TensorPair create_tensor_pair(Engine eg, Pair kv) {
        File[] inputs = (File[]) kv.input;
        Integer[] labels = (Integer[]) kv.label;
        
        Future<Tensor> finput = exec.submit(() -> { return key_transform.transform(eg, to_pixels(inputs)); });
        Future<Tensor> flabel = exec.submit(() -> { return value_transform.transform(eg, labels); });

        Tensor input, label;
        try { input = finput.get(); label = flabel.get(); }
        catch(InterruptedException | ExecutionException e) { 
            throw new RuntimeException(e); 
        }
        
        return new TensorPair(input, label);
    }
    //</editor-fold>
    public synchronized ImageFolder init() {
        if (!ff.inited) img_exec = Executors.newFixedThreadPool(num_threads, daemonThreadFactory);
        ff.init(); 
        return this; 
    }
    
    @Override public ImageFolder shuffle_sort() { ff.shuffle_sort(); return this; }
    @Override public ImageFolder shuffle_swap(double percent) { ff.shuffle_swap(percent); return this; }
    @Override public ImageFolder shuffle_swap() { ff.shuffle_swap(); return this; } 
    
    @Override public TensorPair get(Engine eg) { return create_tensor_pair(eg, ff.container().get(1)); }
    @Override public TensorPair get(Engine eg, int batch) { return create_tensor_pair(eg, ff.container().get(batch)); }
    
    @Override public ImageFolder[] split_clear(int sub_size) { ImageFolder[] subs = split(sub_size); clear(); return subs; }
    @Override public ImageFolder[] split_clear(float percent) { ImageFolder[] subs = split(percent); clear(); return subs; }
    @Override public ImageFolder[] split(float percent) { return split((int)(size() * percent)); }
    @Override public ImageFolder[] split(int sub_size) {
        FileFolder[] ffs = ff.split(sub_size);
        ImageFolder first = new ImageFolder(ffs[0], pixel_transform, num_threads);
        ImageFolder last  = new ImageFolder(ffs[1], pixel_transform, num_threads);
        return new ImageFolder[]{ first, last };
    }

    @Override public ImageFolder[] class_split_clear(float percent) { ImageFolder[] subs = class_split(percent); clear(); return subs; }
    @Override public Map<Integer, Integer> class_sample_num() { return ff.class_sample_num(); }
    @Override public ImageFolder[] class_split(float percent) {
        FileFolder[] ffs = ff.class_split(percent);
        ImageFolder first = new ImageFolder(ffs[0], pixel_transform, num_threads);
        ImageFolder last  = new ImageFolder(ffs[1], pixel_transform, num_threads);
        return new ImageFolder[]{ first, last };
    }
    //</editor-fold>
    
    @Override
    public TensorIter batch_iter() { return new BIter(ff.container().batch_iter());  }
}
