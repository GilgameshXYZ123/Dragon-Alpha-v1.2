/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;

import com.sun.imageio.plugins.jpeg.JPEGImageReader;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class Load 
{
    static { alpha.home(Config.alpha_home); } //size = 1281167
    static Mempool memp = alpha.engine.memp2(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    
    final static ArrayList<File> arr = new ArrayList<>(1 << 20);
    final static ExRandom exr = new ExRandom();
    static {
        File dirs = new File("D:\\virtual-disc-V-data\\Imagenet2012\\ILSVRC2012_img_train");
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
     
    static ExecutorService exec = Executors.newFixedThreadPool(16, daemonThreadFactory);
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
        
        Future[] fts = new Future[batchSize];
        for(int i=0; i<batchSize; i++) {
            final int idx = i;
            fts[i] = exec.submit(()->{
                BufferedImage img = null;
                try { img = cv.imread(f[idx]); }
                catch(Exception e) { 
                    System.err.println(f[idx]);
                    return null;
                }
                
                if(cv.channels(img) != 3) img = cv.to_BGR(img);
                img = cv.reshape(img, 128, 128);
                return cv.pixel(img);
            });
        }
        
        byte[][] pix = new byte[batchSize][];
        try {  for(int i=0; i<batchSize; i++) pix[i] = (byte[]) fts[i].get(); }
        catch(InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }
    
    public static void testSpeed(int nIter, int batchSize) {
        SimpleTimer timer = new SimpleTimer().record();
        int offset = 0;
        
        //only load: 196 ms
        //reshape: 172 ms
        //12 thread + reshape: 139
        
        int length = arr.size() / batchSize * batchSize;
        if(length > nIter) length = nIter;
//        nIter = length;
        
        for(int i=0; i<nIter; i++) {
            test3(batchSize, offset);
            offset += batchSize;
        }
        
        
        
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
        
        long time = dif / nIter;
        System.out.println("avg_time = " + time);
    }
    
    public static void full_load_test(int nIter, int batchSize) {
        SimpleTimer timer = new SimpleTimer().record();
        int offset = 0;
        
        //only load: 196 ms
        //reshape: 172 ms
        //12 thread + reshape: 139
        
        int length = arr.size() / batchSize * batchSize;
        if(length > nIter) length = nIter;
//        nIter = length;
        
        for(int i=0; i<nIter; i++) {
            test3(batchSize, offset);
            offset += batchSize;
        }
        
        
        
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
        
        long time = dif / nIter;
        System.out.println("avg_time = " + time);
    }
    
    public static void test4() {
        String file1 = "D:\\virtual-disc-V-data\\Imagenet2012\\ILSVRC2012_img_train\\n01692333\\n01692333_2556.JPEG";
        String file2 = "C:\\Users\\Tourist\\Desktop\\n01692333_2556.jpg";
        String file3 = "D:\\virtual-disc-V-data\\Imagenet2012\\ILSVRC2012_img_train\\n03126707\\n03126707_17264.jpg";
        
        try
        {
//            Iterator<ImageReader> iter  = ImageIO.getImageReadersBySuffix("jpg");
//            while(iter.hasNext()) System.out.println(iter.next().getClass());
            
            InputStream in = new BufferedInputStream(new FileInputStream(file3));
            BufferedImage img = ImageIO.read(in);
            //BufferedImage img = ImageIO.read(new File(file1));
            
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) 
    {
//        System.out.println(Runtime.getRuntime().availableProcessors());
//        test4();
        testSpeed(3000, 256);
        
        System.out.println(arr.size());
    }
}
