/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ImageFolder;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;

/**
 *
 * @author Gilgamesh
 */
public class Load1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp2(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    
    final static ArrayList<File> arr = new ArrayList<>(1 << 20);
    final static ExRandom exr = new ExRandom();
    static {
        File dirs = new File("E:\\virtual-disc-V-data\\imagenet2012-train");
        for(File dir : dirs.listFiles()) 
            for(File f : dir.listFiles()) arr.add(f);
    }
    
    public static void test1(int batchSize) {
        File[] f = new File[batchSize];
        for(int i=0; i<batchSize; i++) f[i] = arr.get(i);
        
        SimpleTimer timer = new SimpleTimer().record();
        BufferedImage[] img = new BufferedImage[batchSize];
        try {
            for(int i=0; i<batchSize; i++) img[i] = cv.imread(f[i]);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
        
    }
    
    private static final ThreadFactory daemonThreadFactory = (Runnable r) -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    };
     
    static ExecutorService exec = Executors.newFixedThreadPool(8, daemonThreadFactory);
    public static void test2(int batchSize, int offset) {
        File[] f = new File[batchSize];
        for(int i=0; i<batchSize; i++) f[i] = arr.get(i + offset);
        
        SimpleTimer timer = new SimpleTimer().record();
        Future[] fts = new Future[batchSize];
        for(int i=0; i<batchSize; i++) {
            int index = i;
            fts[i] = exec.submit(()->{
                try { return cv.imread(f[index]);  }
                catch(Exception e) { throw new RuntimeException(e); }
            });
        }
        
        BufferedImage[] img = new BufferedImage[batchSize];
        for(int i=0; i<batchSize; i++) {
            try {
                img[i] = (BufferedImage) fts[i].get();
            }
            catch(InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);   
    }
    
     public static void test3(int batchSize, int offset) {
        File[] f = new File[batchSize];
        for(int i=0; i<batchSize; i++) f[i] = arr.get(i + offset);
        
        SimpleTimer timer = new SimpleTimer().record();
        Future[] fts = new Future[batchSize];
        for(int i=0; i<batchSize; i++) {
            int index = i;
            fts[i] = exec.submit(()->{
                try { 
                    BufferedImage img = cv.imread(f[index]);
                    img = cv.reshape(img, 128, 128);
                    return cv.pixel(img);  
                }
                catch(Exception e) { throw new RuntimeException(e); }
            });
        }
        
        byte[][] pixels = new byte[batchSize][];
        for(int i=0; i<batchSize; i++) {
            try { 
                pixels[i] = (byte[]) fts[i].get();
            }
            catch(InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
        
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
    }
    
    public static void main(String[] args) {
        int offset = 0;
        for(int i=0; i<1000; i++) {
            test3(256, offset);
            offset += 256;
        }
    }
}
