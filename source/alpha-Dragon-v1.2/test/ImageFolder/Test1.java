/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ImageFolder;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class Test1 
{
    public static void test1()
    {
        SimpleTimer timer = SimpleTimer.clock();
        
        //count == 100: 142
        //count == 200: 267
        //count == 300: 388
        //count == 400: 514
        //count == 500: 638
        //count == 600: 22484
        //String: list():
        //count = 600: 724
        //count = 700: 44565 
        //System.gc(): 2802
        //System.gc(100):  850
        //
        
        int count = 0;
        int num = 0;
            
        Map<String, Integer> map = new HashMap<>();
        
        File dirs = new File("E:\\virtual-disc-V-data\\imagenet2012-train");
        for(File dir : dirs.listFiles())
        {
            System.out.println(count + " : " + dir.getName()); count++;
            for(String f : dir.list())  {
                String id = f.substring(f.indexOf('-') + 1, f.indexOf('.'));
                if(map.get(id) == null) map.put(id, 0);
                int value = map.get(id);
                map.put(id, value + 1);
            }
            
            System.gc();
            if(count == 800) break;
        }
        
        System.out.println(map.size());
        System.out.println("count = " + count);
        System.out.println("num = " + num);
        
        long time = timer.record().timeStamp_dif_millis();
        System.out.println("time = " + time);
    }
    
    public static void main(String[] args)
    {
        test1();
    }
}
