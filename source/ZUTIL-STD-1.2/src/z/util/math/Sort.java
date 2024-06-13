/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math;

import java.util.ArrayList;
import java.util.Comparator;
import z.util.math.vector.Vector;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import z.util.concurrent.BinarySemaphore;
import z.util.lang.Lang;
import static z.util.lang.Lang.MB_BIT;
import z.util.math.vector.Vector.MaxMin;

/**
 * @author dell
 */
@SuppressWarnings("empty-statement")
public final class Sort
{
    public static final int INSERT_SORT_THRESHOLD = 4;
    public static final int SELECT_SORT_THRESHOLD = 8;
    public static final int SHELL_SORT_THRESHOLD = 32;
    public static final int QUICK_SORT_THRESHOLD = 186;
    public static final int COUNTING_SORT_RANGE_INT = (MB_BIT) / Integer.SIZE;
    public static final int TIM_SORT_SEGMENT_NUM = 48;
    public static final int SINGLE_THREAD_THRESHOLD = 130000;
    public static final int MULTI_THREAD_LEAF = 500000;
    
    private Sort() {}
     
    //<editor-fold defaultstate="collapsed" desc="multi-thread-support">
    static final ThreadFactory daemonThreadFactory = (Runnable r) -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    };
    static final ExecutorService exec = Executors.newFixedThreadPool(16, daemonThreadFactory);
    static final BinarySemaphore mutex = new BinarySemaphore();
    
    static void sync(ArrayList<Future<?>> fts) {
        try { for(Future<?> ft : fts) ft.get(); }
        catch(InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }
    
    public static void shutDown() {
        mutex.P();
        if(!exec.isShutdown()) exec.shutdown();
        mutex.V();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sort: Comparable">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static void insert_sort(Comparable... a) { insert_sort(a, 0, a.length - 1); }
    public static void insert_sort(Comparable[] a, int low, int high) {
        for (int i = low + 1; i <= high; i++) {//a[i] < a[j]
            int j; Comparable t;
            for (t = a[low], a[low] = a[i], j = i - 1; a[low].compareTo(a[j]) < 0; j--) a[j + 1] = a[j];
            if (j != low) { a[j + 1] = a[low]; a[low] = t; continue; }
            if (a[low].compareTo(t) < 0) a[low + 1] = t;//k==a[0]
            else { a[low + 1] = a[low]; a[low] = t;}
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static void select_sort(Comparable[] a) { select_sort(a, 0, a.length - 1); }
    public static void select_sort(Comparable[] a, int low, int high) {
        for(int i = low; i < high; i++)
        for(int j = i+1; j <= high; j++)
            if(a[j].compareTo(a[i]) < 0) { Comparable p = a[i]; a[i] = a[j]; a[j] = p;}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort"> 
    public static void shell_sort(Comparable[] a) { shell_sort(a, 0, a.length - 1); }
    public static void shell_sort(Comparable[] a, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; h = h * 3 + 1);
        for(; h > 0; h /= 3) {
            for(int i = h + low; i <= high; i++)
            for(int j = i, p; (j >= h) && a[j].compareTo(a[p = j - h]) < 0; j = p) {
                Comparable t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="binaryHeap_sort">
    public static void heap_sort(Comparable... a) { heap_sort(a, 0, a.length - 1); }
    public static void heap_sort(Comparable[] a, int low ,int high) {
        for(int i = (low + high) >> 1; i >= low; i--)//create max-heap
        for(int p = i, left = (p << 1) + 1; left <= high; left = (p << 1) + 1) {
            int max = (a[p].compareTo(a[left]) >= 0 ? p : left);
            if(++left <= high) max = (a[max].compareTo(a[left]) >= 0 ? max : left);
            if(max == p) break;
            Comparable t = a[max]; a[max] = a[p]; a[p] = t;
            p = max;
        }
        
        for(int i = high; i > low; ) {//exchange A[0] and A[i], then max-down().
            Comparable t = a[low]; a[low] = a[i]; a[i] = t;
            --i;
            for(int p = 0, left = 1; left <= i; left = (p << 1) + 1) {
                int max = (a[p].compareTo(a[left]) >= 0 ? p : left);
                if(++left <= i) max = (a[max].compareTo(a[left]) >= 0? max : left);
                if(max == p) break;
                t = a[max]; a[max] = a[p]; a[p] = t;
                p = max;
            }
        }
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="quick_sort">
    public static int quick_part(Comparable[] a, int low, int high) {
        Comparable t, p = a[low];
        while (low < high) {
            while (low < high && a[high].compareTo(p) >= 0) high--;
            t = a[low]; a[low] = a[high]; a[high] = t;
            while (low < high && a[low].compareTo(p) <= 0) low++;
            t = a[low]; a[low] = a[high]; a[high] = t;
        }
        return low;
    }
    
    public static void quick_sort(Comparable[] a) { quick_sort(a, 0, a.length - 1); }
    public static void quick_sort(Comparable[] a, int low, int high) {
        if(low <= high) return;
        int idx = Lang.exr().nextInt(low,high);
        Comparable t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        int p = Sort.quick_part(a, low, high);
        quick_sort(a, low, p - 1);
        quick_sort(a, p + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="threeQuick_sort">
    public static long three_part(Comparable[] a, int low, int high) {
        Comparable p = a[low],t;
        for(int k = low + 1; k <= high;)  {
            while(k <= high) {
                if(a[k].compareTo(p)<0) { t = a[low]; a[low] = a[k]; a[k] = t; low++; }
                else if(a[k].compareTo(p) > 0) break;
                k++;
            }
            while(k <= high) {
                if(a[high].compareTo(p) > 0) high--;
                else if(a[high].compareTo(p) == 0) {
                    t = a[k]; a[k] = a[high]; a[high] = t;
                    k++; high--; break;
                }
                else {
                    t = a[low]; a[low] = a[high]; a[high] = a[k]; a[k] = t;
                    low++; k++; high--; break;
                }
            }
        }
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
    
    public static void threeQuick_sort(Comparable[] a) { threeQuick_sort(a, 0, a.length - 1); }
    public static void threeQuick_sort(Comparable[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low,high);
        Comparable t  =a[low]; a[low] = a[idx]; a[idx] = t;
        
        long p = three_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        threeQuick_sort(a, low, p0 - 1);
        threeQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
      public static <T> long dualPivot_part(Comparable[] a, int low ,int high) {
        Comparable<T> t;
        if(a[low].compareTo(a[high]) >= 0) { t = a[low]; a[low] = a[high]; a[high] = t;}
        
        Comparable p1 = a[low], p2 = a[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if (a[k].compareTo(p1) < 0) { t = a[i]; a[i] = a[k]; a[k] = t; i++;}
                else if(a[k].compareTo(p2) >= 0) break;
                k++;
            }
            while(k<=j) {
                if (a[j].compareTo(p2) > 0) j--;
                else if (a[j].compareTo(p1) >= 0 && a[j].compareTo(p2) <= 0) {
                    t = a[j]; a[j] = a[k]; a[k] = t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i]; a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        
        i--; j++;
        t = a[low];  a[low]  = a[i]; a[i] = t;
        t = a[high]; a[high] = a[j]; a[j] = t;
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
    
    public static void dualPivotQuick_sort(Comparable[] a) { dualPivotQuick_sort(a, 0, a.length - 1); }
    public static void dualPivotQuick_sort(Comparable[] a, int low, int high) {
        if(low >= high) return;
        int idx; Comparable t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        dualPivotQuick_sort(a, low, p0 - 1);
        dualPivotQuick_sort(a, p0 + 1, p1 - 1);
        dualPivotQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="merge_sort">
    public static void merge(Comparable[] a, int low, int mid, int high)  {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { insert_sort(a, low, high); return; }
        
        Comparable[] b = new Comparable[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) a[end--] = (a[i].compareTo(b[j]) > 0? a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    public static void merge_sort(Comparable[] a) { merge_sort(a, 0, a.length - 1); }
    public static void merge_sort(Comparable[] a, int low, int high) {
        if(low >= high) return;
        int mid = (low + high) >> 1;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    public static void tim_merge(int[] edge, int len, Comparable[] a) {//len>=1
        int[] st = new int[4];//if there are three segments in stack, then combine y with smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(a, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static boolean tim_sort(Comparable... a) { return tim_sort(a, 0, a.length - 1); }
    public static boolean tim_sort(Comparable[] a, int low, int high) {
        int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i + 1]]
        for(int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if(a[k].compareTo(a[k + 1]) <= 0) while((++k < high) && a[k].compareTo(a[k + 1]) <= 0);
            else {
                while((++k < high) && a[k].compareTo(a[k + 1]) >= 0);
                for(int start = edge[index],end = k; start < end; start++, end--) {
                    Comparable t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high+1;
            tim_merge(edge, index, a);
            return true;
        }
        return false;
    } 
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi_thread_sort">
    static void msort(ArrayList<Future<?>> fts, Comparable[] a, int low, int high) {
        int len = high - low + 1, idx; Comparable t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL),p1 = (int) (p >> 32);

        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, low, p0-1); return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); }
            return;
        }

        msort(fts, a, low, p0 - 1);
        msort(fts, a, p0 + 1, p1 - 1);
        msort(fts, a, p1 + 1, high);
    }
    //</editor-fold>
    public static void sort(Comparable[] a) { sort(a, 0, a.length - 1); }
    public static void sort(Comparable[] a, int low, int high) {
        int len = high - low + 1, idx; Comparable t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, low, high); return; }//checked
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, low, high); return; }//checked
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom ran = Lang.exr();
            idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
            idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
            long p = dualPivot_part(a, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        if (tim_sort(a, low, high)) return;
        if (len > MULTI_THREAD_LEAF) {
            ArrayList<Future<?>> fts = new ArrayList<>(4); 
            msort(fts, a, low, high);
            sync(fts); return;
        }
       
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, low, p0 - 1);
        sort(a, p0 + 1, p1 - 1);
        sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sort: object + cmp">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static <T> void insert_sort(T[] a, Comparator<T> cmp) { insert_sort(a, cmp, 0, a.length - 1); }
    public static <T> void insert_sort(T[] a, Comparator<T> cmp, int low, int high) {
        for(int i = low + 1; i <= high; i++) {
            int j; T t;//a[i] < a[j]
            for(t = a[low], a[low] = a[i], j = i - 1; cmp.compare(a[low], a[j]) < 0; j--) a[j + 1] = a[j];
            if(j != low) { a[j + 1] = a[low]; a[low] = t; continue; }
            if(cmp.compare(a[low], t) < 0) a[low + 1] = t;//k == a[0]
            else { a[low + 1] = a[low]; a[low] = t;}
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static <T> void select_sort(T[] a, Comparator<T> cmp) { select_sort(a, cmp, 0, a.length - 1); }
    public static <T> void select_sort(T[] a, Comparator<T> cmp, int low, int high) {
        for(int i = low; i < high; i++)
        for(int j = i+1; j <= high; j++)
            if(cmp.compare(a[j], a[i]) < 0) { T p = a[i]; a[i] = a[j]; a[j] = p; }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static <T> void shell_sort(T[] a, Comparator<T> cmp) { shell_sort(a, cmp, 0, a.length - 1); }
    public static <T> void shell_sort(T[] a, Comparator<T> cmp, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; h = h * 3 + 1);
        for(; h > 0; h /= 3) {
            for(int i = h + low; i <= high; i++)
            for(int j = i, p; (j >= h) && cmp.compare(a[j], a[p = j - h]) < 0; j = p) { 
                T t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static <T> long dualPivot_part(T[] a, Comparator<T> cmp, int low ,int high) {
        T t; if(cmp.compare(a[low], a[high]) >= 0) { t = a[low]; a[low] = a[high]; a[high] = t; }
        
        T p1 = a[low], p2 = a[high];
        int i = low + 1, k = low + 1, j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if(cmp.compare(a[k], p1) < 0) { t = a[i]; a[i] = a[k]; a[k] = t; i++; }
                else if(cmp.compare(a[k], p2) >= 0) break;
                k++;
            }
            while (k <= j) {
                if (cmp.compare(a[j], p2) > 0) j--;
                else if (cmp.compare(a[j], p1) >= 0 && cmp.compare(a[j], p2) <= 0) {
                    t = a[j]; a[j] = a[k]; a[k] = t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i]; a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        t = a[low];  a[low]  = a[i]; a[i] = t;
        t = a[high]; a[high] = a[j]; a[j] = t;
        return ((long)i & 0xFFFFFFFFL) | (((long)j << 32) & 0xFFFFFFFF00000000L);
    }
     
    public static <T> void dualPivotQuick_sort(T[] a, Comparator<T> cmp) { dualPivotQuick_sort(a, cmp, 0, a.length - 1); }
    public static <T> void dualPivotQuick_sort(T[] a, Comparator<T> cmp, int low, int high) {
        if(low >= high) return;
        int idx; T t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high);
        t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        idx = ran.nextInt(low,high);
        t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, cmp, low, high);
        int p0 = Num.low32(p),p1 = Num.high32(p);
        dualPivotQuick_sort(a, cmp, low, p0 - 1);
        dualPivotQuick_sort(a, cmp, p0 + 1, p1 - 1);
        dualPivotQuick_sort(a, cmp, p1 + 1, high);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="tim_sort"> 
    public static void merge(Object[] a, Comparator cmp, int low, int mid, int high) {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { insert_sort(a, cmp, low, high); return; }
         
        Object[] b = new Object[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) a[end--] = (cmp.compare(a[i], b[j]) > 0 ? a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    static <T> void tim_merge(int[] edge, int len, T[] a, Comparator<T> cmp) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y with smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(a, cmp, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, cmp, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, cmp, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static <T> boolean tim_sort(T[] a, Comparator<T> cmp) { return tim_sort(a, cmp, 0, a.length - 1); }
    public static <T> boolean tim_sort(T[] a, Comparator<T> cmp, int low, int high) {
        int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i + 1]]
        for(int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if(cmp.compare(a[k], a[k+1]) <= 0) while((++k < high) && cmp.compare(a[k], a[k + 1]) <= 0);
            else {
                while((++k < high) && cmp.compare(a[k], a[k + 1]) >= 0);
                for(int start = edge[index], end = k; start < end; start++, end--) {
                    T t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM) {
            edge[++index] = high + 1;
            tim_merge(edge, index, a, cmp);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi_thread_sort">
    static <T> void msort(ArrayList<Future<?>> fts, T[] a, Comparator<T> cmp, int low, int high) {
        int len = high - low + 1, idx; T t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, cmp, low, high);
        int p0 = (int) (p & 0x000000ffffffffL),p1 = (int) (p >> 32);
        
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, cmp, low, p0 - 1);
            sort(a, cmp, p0 + 1, p1 - 1);
            sort(a, cmp, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, cmp, low, p0 - 1); return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, cmp, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, cmp, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); }
            return;
        }

        msort(fts, a, cmp, low, p0 - 1);
        msort(fts, a, cmp, p0 + 1, p1 - 1);
        msort(fts, a, cmp, p1 + 1, high);
    }
    //</editor-fold>
    public static <T> void sort(T[] a, Comparator<T> cmp) { sort(a, cmp, 0, a.length - 1); }
    public static <T> void sort(T[] a, Comparator<T> cmp, int low, int high) {
        int len = high - low + 1, idx; T t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, cmp, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, cmp, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD)  {
            ExRandom ran = Lang.exr();
            idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
            idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
            long p = dualPivot_part(a, cmp, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(a, cmp, low, p0 - 1);
            sort(a, cmp, p0 + 1, p1 - 1);
            sort(a, cmp, p1 + 1, high);
            return;
        }
        if (tim_sort(a, cmp, low, high)) return;
        if (len > MULTI_THREAD_LEAF) { 
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, a, cmp, low, high);
            sync(fts); return;
        }
       
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, cmp, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, cmp, low, p0 - 1);
        sort(a, cmp, p0 + 1, p1 - 1);
        sort(a, cmp, p1 + 1, high);
    }
    //</editor-fold> 
    
    //<editor-fold defaultstate="collapsed" desc="sort: byte">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static void insert_sort(byte[] a, int low) { insert_sort(a, 0, a.length - 1); }
    public static void insert_sort(byte[] a, int low, int high) {
        for(int i = low + 1; i <= high; i++) {
            int j; byte t;//a[i] < a[j]
            for(t = a[low], a[low] = a[i], j = i - 1; a[low] < a[j]; j--) a[j + 1] = a[j];
            if(j != low) { a[j + 1] = a[low]; a[low] = t; continue; }
            if(a[low] < t) a[low + 1] = t;//k == a[0]
            else { a[low + 1] = a[low]; a[low] = t; }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static void select_sort(byte... a) { select_sort(a, 0, a.length - 1); }
    public static void select_sort(byte[] a, int low, int high) {
        for(int i = low; i < high; i++)
        for(int j = i+1; j <= high; j++)
            if(a[j] < a[i]) { byte t = a[i]; a[i] = a[j]; a[j] = t;}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static void shell_sort(byte... a) { shell_sort(a, 0, a.length - 1); }
    public static void shell_sort(byte[] a, int low, int high) {
        int h = 1; for (int top = (high - low + 1) / 3; h < top; h = h * 3 + 1);
        for (; h > 0; h /= 3) {
            for(int i = h + low; i <= high; i++)
            for(int j = i, p; (j >= h) && (a[j] < a[p = j - h]); j = p) {
                byte t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="quick_sort">
    public static int quick_part(byte[] a, int low, int high) {
        byte t, p = a[low];
        while (low < high) {
            while (low < high && a[high] >= p) high--;
            t = a[low]; a[low] = a[high]; a[high] = t;
            while (low < high && a[low] <= p) low++;
            t = a[low]; a[low] = a[high]; a[high] = t;
        }
        return low;
    }
     
    public static void quick_sort(byte... a) { quick_sort(a, 0, a.length - 1); }
    public static void quick_sort(byte[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low, high);
        byte t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        int p = Sort.quick_part(a, low, high);
        quick_sort(a, low, p - 1);
        quick_sort(a, p + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static long dualPivot_part(byte[] a, int low ,int high) {
        byte t; if(a[low] > a[high]) { t = a[low]; a[low] = a[high]; a[high] = t;}
        
        byte p1 = a[low], p2 = a[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if (a[k ] < p1) { t = a[i]; a[i] = a[k]; a[k] = t; i++;}
                else if (a[k] >= p2) break;
                k++;
            }
            while (k <= j) {
                if (a[j] > p2) j--;
                else if (a[j] >= p1 && a[j] <= p2)  {
                    t = a[j]; a[j] = a[k]; a[k] = t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i];a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        t = a[low]; a[low] = a[i]; a[i]=t;
        t = a[high]; a[high] = a[j]; a[j]=t;
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
      
    public static void dualPivotQuick_sort(byte... a) { dualPivotQuick_sort(a, 0, a.length - 1); }
    public static void dualPivotQuick_sort(byte[] a, int low, int high) {
        if(low >= high) return;
        int idx; byte t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high);
        t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        idx = ran.nextInt(low,high);
        t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        dualPivotQuick_sort(a, low, p0 - 1);
        dualPivotQuick_sort(a, p0 + 1, p1 - 1);
        dualPivotQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="merge_sort">
    public static void merge(byte[] a, int low, int mid, int high)  {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { insert_sort(a, low, high); return; }
        
        byte[] b = new byte[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) a[end--] = (a[i] > b[j] ? a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    public static void merge_sort(byte... a) { merge_sort(a, 0, a.length - 1); }
    public static void merge_sort(byte[] a, int low, int high) {
        if(low >= high) return;
        int mid = (low + high) >> 1;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    static void tim_merge(int[] edge, int len, byte[] a) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y withs smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(a, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static boolean tim_sort(byte[] a) { return tim_sort(a, 0, a.length - 1); }
    public static boolean tim_sort(byte[] a, int low, int high) {
        int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i + 1]]
        for(int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if(a[k] <= a[k + 1]) while((++k < high) && (a[k] <= a[k + 1]));
            else {
                while((++k < high) && (a[k] >= a[k + 1]));
                for(int start = edge[index],end = k; start < end; start++, end--) {
                    byte t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, a);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="counting_sort">
    public static boolean counting_sort(byte... a) { return counting_sort(a, 0, a.length - 1); }
    public static boolean counting_sort(byte[] a, int low, int high) {
        int threshold = COUNTING_SORT_RANGE_INT;
        int free_memory = (int) (Runtime.getRuntime().freeMemory() >> 1);
        if(threshold > free_memory) threshold = free_memory;
        
        MaxMin<Byte> mm = Vector.maxMin(a, low, high, threshold);
        if(mm == null) return false;
        
        int min = mm.min(), dif = mm.max() - min + 1; 
        if(dif > threshold) return false;
        
        int[] h = new int[dif];
        for (int i = low; i <= high; i++) h[a[i] - min]++;
        for (int i = 0, index = low; i < dif; i++)
        for (int j = 0; j < h[i]; j++) a[index++] = (byte) (min + i);
        return true;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thread-sort">
    static void msort(ArrayList<Future<?>> fts, byte[] a, int low, int high) {
        int len = high - low + 1, idx; byte t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        
        if(len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        
        if(len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, low, p0 - 1); return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); } 
            return;
        }

        msort(fts, a, low, p0 - 1);
        msort(fts, a, p0 + 1, p1 - 1);
        msort(fts, a, p1 + 1, high);
    }
    //</editor-fold>
    public static void sort(byte... a) { sort(a, 0, a.length - 1); }
    public static void sort(byte[] a, int low, int high) {
        int len = high - low + 1, idx; byte t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom ran = Lang.exr();
            idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
            idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
            long p = dualPivot_part(a, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        if (tim_sort(a, low, high)) return;
        if (counting_sort(a, low, high)) return;
        if (len > MULTI_THREAD_LEAF) { 
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, a, low, high);
            sync(fts); return; 
        }
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, low, p0 - 1);
        sort(a, p0 + 1, p1 - 1);
        sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sort: char">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static void insert_sort(char[] a, int low) { insert_sort(a, 0, a.length - 1); }
    public static void insert_sort(char[] a, int low, int high) {
        for (int i = low + 1; i <= high; i++) {
            int j; char t;//a[i] < a[j]
            for (t = a[low], a[low] = a[i], j = i - 1; a[low] < a[j]; j--) a[j + 1] = a[j];
            if (j != low) { a[j + 1] = a[low]; a[low] = t; continue; }
            if (a[low] < t) a[low + 1] = t;//k == a[0]
            else { a[low + 1] = a[low]; a[low] = t; }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static void select_sort(char... a) { select_sort(a, 0, a.length - 1); }
    public static void select_sort(char[] a, int low, int high) {
        for (int i = low; i < high; i++)
        for (int j = i+1; j <= high; j++)
            if(a[j] < a[i]) { char t = a[i]; a[i] = a[j]; a[j] = t;}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static void shell_sort(char... a) { shell_sort(a, 0, a.length - 1); }
    public static void shell_sort(char[] a, int low, int high) {
        int h = 1; for (int top = (high - low + 1) / 3; h < top; h = h * 3 + 1);
        for (; h > 0; h /= 3) {
            for (int i = h + low; i <= high; i++)
            for (int j = i, p; (j >= h) && (a[j] < a[p = j - h]); j = p) {
                char t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="quick_sort">
    public static int quick_part(char[] a, int low, int high) {
        char t, p = a[low];
        while (low < high) {
            while (low < high && a[high] >= p) high--;
            t = a[low]; a[low] = a[high]; a[high] = t;
            while (low < high && a[low] <= p) low++;
            t = a[low]; a[low] = a[high]; a[high] = t;
        }
        return low;
    }
     
    public static void quick_sort(char... a) { quick_sort(a, 0, a.length - 1); }
    public static void quick_sort(char[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low, high);
        char t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        int p = Sort.quick_part(a, low, high);
        quick_sort(a, low, p - 1);
        quick_sort(a, p + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static long dualPivot_part(char[] a, int low ,int high) {
        char t; if(a[low] > a[high]) { t = a[low]; a[low] = a[high]; a[high] = t;}
        
        char p1 = a[low], p2 = a[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if (a[k ] < p1) { t = a[i]; a[i] = a[k]; a[k] = t; i++;}
                else if (a[k] >= p2) break;
                k++;
            }
            while (k <= j) {
                if (a[j] > p2) j--;
                else if (a[j] >= p1 && a[j] <= p2)  {
                    t = a[j]; a[j] = a[k]; a[k] = t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i];a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        t = a[low]; a[low] = a[i]; a[i] = t;
        t = a[high]; a[high] = a[j]; a[j] = t;
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
      
    public static void dualPivotQuick_sort(char... a) { dualPivotQuick_sort(a, 0, a.length - 1); }
    public static void dualPivotQuick_sort(char[] a, int low, int high) {
        if(low >= high) return;
        int idx; char t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        dualPivotQuick_sort(a, low, p0 - 1);
        dualPivotQuick_sort(a, p0 + 1, p1 - 1);
        dualPivotQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="merge_sort">
    public static void merge(char[] a, int low, int mid, int high)  {
        if (high - low + 1 <= INSERT_SORT_THRESHOLD) { insert_sort(a, low, high); return; }
        
        char[] b = new char[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while (i >= low && j >= 0) a[end--] = (a[i] > b[j] ? a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    public static void merge_sort(char... a) { merge_sort(a, 0, a.length - 1); }
    public static void merge_sort(char[] a, int low, int high) {
        if(low >= high) return;
        int mid = (low + high) >> 1;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    static void tim_merge(int[] edge, int len, char[] a) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y withs smaller(z,x)
        for (int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if (index == 4) {
                if (st[3] - st[2] < st[1] - st[0]) merge(a, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static boolean tim_sort(char[] a) { return tim_sort(a, 0, a.length - 1); }
    public static boolean tim_sort(char[] a, int low, int high) {
        int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i + 1]]
        for (int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if (a[k] <= a[k + 1]) while((++k < high) && (a[k] <= a[k + 1]));
            else {
                while ((++k < high) && (a[k] >= a[k + 1]));
                for (int start = edge[index],end = k; start < end; start++, end--) {
                    char t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, a);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="counting_sort">
    public static boolean counting_sort(char... a) { return counting_sort(a, 0, a.length - 1); }
    public static boolean counting_sort(char[] a, int low, int high) {
        int threshold = COUNTING_SORT_RANGE_INT;
        int free_memory = (int) (Runtime.getRuntime().freeMemory() >> 1);
        if(threshold > free_memory) threshold = free_memory;
        
        MaxMin<Character> mm = Vector.maxMin(a, low, high, threshold);
        if(mm == null) return false;
        
        char min = mm.min(); int dif = mm.max() - min + 1; 
        if(dif > threshold) return false;
        
        int[] h = new int[dif];
        for (int i = low; i <= high; i++) h[a[i] - min]++;
        for (int i = 0, index = low; i < dif; i++)
        for (int j = 0; j < h[i]; j++) a[index++] = (char) (min + i);
        return true;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thread-sort">
    static void msort(ArrayList<Future<?>> fts, char[] a, int low, int high) {
        int len = high - low + 1, idx; char t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, low, p0 - 1); return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); } 
            return;
        }

        msort(fts, a, low, p0 - 1);
        msort(fts, a, p0 + 1, p1 - 1);
        msort(fts, a, p1 + 1, high);
    }
    //</editor-fold>
    public static void sort(char... a) { sort(a, 0, a.length - 1); }
    public static void sort(char[] a, int low, int high) {
        int len = high - low + 1, idx; char t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom ran = Lang.exr();
            idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
            idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
            long p = dualPivot_part(a, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        if (tim_sort(a, low, high)) return;
        if (counting_sort(a, low, high)) return;
        if (len > MULTI_THREAD_LEAF) { 
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, a, low, high);
            sync(fts); return; 
        }
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, low, p0 - 1);
        sort(a, p0 + 1, p1 - 1);
        sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sort: short">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static void insert_sort(short[] a, int low) { insert_sort(a, 0, a.length - 1); }
    public static void insert_sort(short[] a, int low, int high) {
        for (int i = low + 1; i <= high; i++) {
            int j; short t;//a[i] < a[j]
            for (t = a[low], a[low] = a[i], j = i - 1; a[low] < a[j]; j--) a[j + 1] = a[j];
            if (j != low) { a[j + 1] = a[low]; a[low] = t; continue; }
            if (a[low] < t) a[low + 1] = t;//k == a[0]
            else { a[low + 1] = a[low]; a[low] = t; }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static void select_sort(short... a) { select_sort(a, 0, a.length - 1); }
    public static void select_sort(short[] a, int low, int high) {
        for (int i = low; i < high; i++)
        for (int j = i+1; j <= high; j++)
            if(a[j] < a[i]) { short t = a[i]; a[i] = a[j]; a[j] = t;}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static void shell_sort(short... a) { shell_sort(a, 0, a.length - 1); }
    public static void shell_sort(short[] a, int low, int high) {
        int h = 1; for (int top = (high - low + 1) / 3; h < top; h = h * 3 + 1);
        for (; h > 0; h /= 3) {
            for (int i = h + low; i <= high; i++)
            for (int j = i, p; (j >= h) && (a[j] < a[p = j - h]); j = p) {
                short t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static long dualPivot_part(short[] a, int low ,int high) {
        short t; if(a[low] > a[high]) { t = a[low]; a[low] = a[high]; a[high] = t;}
        
        short p1 = a[low], p2 = a[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if (a[k] < p1) { t = a[i]; a[i] = a[k]; a[k] = t; i++;}
                else if (a[k] >= p2) break;
                k++;
            }
            while (k <= j) {
                if (a[j] > p2) j--;
                else if (a[j] >= p1 && a[j] <= p2)  {
                    t = a[j]; a[j] = a[k]; a[k] = t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i];a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        t = a[low]; a[low] = a[i]; a[i] = t;
        t = a[high]; a[high] = a[j]; a[j] = t;
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="merge_sort">
    public static void merge(short[] a, int low, int mid, int high)  {
        if (high - low + 1 <= INSERT_SORT_THRESHOLD) { insert_sort(a, low, high); return; }
        
        short[] b = new short[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while (i >= low && j >= 0) a[end--] = (a[i] > b[j] ? a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    public static void merge_sort(short... a) { merge_sort(a, 0, a.length - 1); }
    public static void merge_sort(short[] a, int low, int high) {
        if(low >= high) return;
        int mid = (low + high) >> 1;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    static void tim_merge(int[] edge, int len, short[] a) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y withs smaller(z,x)
        for (int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if (index == 4) {
                if (st[3] - st[2] < st[1] - st[0]) merge(a, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static boolean tim_sort(short... a) { return tim_sort(a, 0, a.length - 1); }
    public static boolean tim_sort(short[] a, int low, int high) {
        int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i + 1]]
        for (int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if (a[k] <= a[k + 1]) while((++k < high) && (a[k] <= a[k + 1]));
            else {
                while ((++k < high) && (a[k] >= a[k + 1]));
                for (int start = edge[index],end = k; start < end; start++, end--) {
                    short t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, a);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="counting_sort">
    public static boolean counting_sort(short... a) { return counting_sort(a, 0, a.length - 1); }
    public static boolean counting_sort(short[] a, int low, int high) {
        int threshold = COUNTING_SORT_RANGE_INT;
        int free_memory = (int) (Runtime.getRuntime().freeMemory() >> 1);
        if(threshold > free_memory) threshold = free_memory;
        
        MaxMin<Short> mm = Vector.maxMin(a, low, high, threshold);
        if(mm == null) return false;
        
        short min = mm.min(); int dif = mm.max() - min + 1; 
        if(dif > threshold) return false;
        
        int[] h = new int[dif];
        for (int i = low; i <= high; i++) h[a[i] - min]++;
        for (int i = 0, index = low; i < dif; i++)
        for (int j = 0; j < h[i]; j++) a[index++] = (short) (min + i);
        return true;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thread-sort">
    static void msort(ArrayList<Future<?>> fts, short[] a, int low, int high) {
        int len = high - low + 1, idx; short t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, low, p0 - 1); return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); } 
            return;
        }

        msort(fts, a, low, p0 - 1);
        msort(fts, a, p0 + 1, p1 - 1);
        msort(fts, a, p1 + 1, high);
    }
    //</editor-fold>
    public static void sort(short... a) { sort(a, 0, a.length - 1); }
    public static void sort(short[] a, int low, int high) {
        int len = high - low + 1, idx; short t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom ran = Lang.exr();
            idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
            idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
            long p = dualPivot_part(a, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        if (tim_sort(a, low, high)) return;
        if (counting_sort(a, low, high)) return;
        if (len > MULTI_THREAD_LEAF) { 
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, a, low, high);
            sync(fts); return; 
        }
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, low, p0 - 1);
        sort(a, p0 + 1, p1 - 1);
        sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sort: int">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static void insert_sort(int[] a, int low) { insert_sort(a, 0, a.length - 1); }
    public static void insert_sort(int[] a, int low, int high) {
        for(int i = low + 1; i <= high; i++) {
            int j, t;//a[i] < a[j]
            for(t = a[low], a[low] = a[i], j = i - 1; a[low] < a[j]; j--) a[j + 1] = a[j];
            if(j != low) { a[j + 1] = a[low]; a[low] = t; continue; }
            if(a[low] < t) a[low + 1] = t;//k == a[0]
            else { a[low + 1] = a[low]; a[low] = t; }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static void select_sort(int... a) { select_sort(a, 0, a.length - 1); }
    public static void select_sort(int[] a, int low, int high) {
        for(int i = low; i < high; i++)
        for(int j = i+1; j <= high; j++)
            if(a[j] < a[i]) { int t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static void shell_sort(int... a) { shell_sort(a, 0, a.length - 1); }
    public static void shell_sort(int[] a, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; h = h * 3 + 1);
        for(; h > 0; h /= 3) {
            for(int i = h + low; i <= high; i++)
            for(int j = i, p; (j >= h) && (a[j] < a[p = j - h]); j = p) {
                int t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="binaryHeap_sort">
    public static void heap_sort(int... a) { heap_sort(a, 0, a.length - 1); }
    public static void heap_sort(int[] a, int low ,int high) {
        int p,left, i;
        for(i = (low+high) >> 1; i >= low; i--)//create max-heap
        for(p = i, left = (p << 1) + 1; left <= high; left = (p << 1) + 1) {
            int max = (a[p] >= a[left] ? p : left);
            if(++left <= high) max = (a[max] >= a[left] ? max : left);
            if(max==p) break;
            int t = a[max]; a[max] = a[p]; a[p] = t;
            p=max;
        }
        
        for(i = high; i > low;) {//exchange A[0] and A[i], then max-down().
            int t = a[low]; a[low] = a[i]; a[i] = t;
            --i;
            for(p = 0, left = 1; left <= i; left = (p << 1) + 1) {
                int max = (a[p] >= a[left] ? p : left);
                if(++left <= i) max = (a[max] >= a[left] ? max : left);
                if(max == p) break;
                t = a[max]; a[max] = a[p]; a[p] = t;
                p=max;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="quick_sort">
    public static int quick_part(int[] a, int low, int high) {
        int t, p = a[low];
        while (low < high) {
            while (low < high && a[high] >= p) high--;
            t = a[low]; a[low] = a[high]; a[high] = t;
            while (low < high && a[low] <= p) low++;
            t = a[low]; a[low] = a[high]; a[high] = t;
        }
        return low;
    }
     
    public static void quick_sort(int... a) { quick_sort(a, 0, a.length - 1); }
    public static void quick_sort(int[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low, high);
        int t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        int p = Sort.quick_part(a, low, high);
        quick_sort(a, low, p - 1);
        quick_sort(a, p + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="threeQuick_sort">
    public static long three_part(int[] a, int low, int high) {
        int p = a[low], t;
        for(int k = low + 1; k <= high; ) {
            while(k <= high) {
                if(a[k] < p) { t = a[low]; a[low] = a[k]; a[k] = t; low++; }
                else if(a[k] > p) break;
                k++;
            }
            while(k <= high) {
                if(a[high] > p) high--;
                else if(a[high] == p)  {
                    t = a[k]; a[k] = a[high]; a[high] = t;
                    k++; high--; break;
                }
                else {
                    t = a[low]; a[low] = a[high]; a[high] = a[k]; a[k] = t;
                    low++; k++; high--; break;
                }
            }
        }
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
    
    public static void threeQuick_sort(int... a) { threeQuick_sort(a, 0, a.length - 1); }
    public static void threeQuick_sort(int[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low,high);
        int t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        long p = three_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        threeQuick_sort(a, low, p0 - 1);
        threeQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static long dualPivot_part(int[] a, int low ,int high) {
        int t; if(a[low] > a[high]) { t = a[low]; a[low] = a[high]; a[high] = t;}
        
        int p1 = a[low], p2 = a[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while(k <= j) {
            while(k <= j) {
                if(a[k ] < p1) { t = a[i]; a[i] = a[k]; a[k] = t; i++;}
                else if(a[k] >= p2) break;
                k++;
            }
            while(k <= j) {
                if(a[j] > p2) j--;
                else if(a[j] >= p1&&a[j] <= p2)  {
                    t = a[j]; a[j] = a[k]; a[k] = t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i];a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        t = a[low]; a[low] = a[i]; a[i]=t;
        t = a[high]; a[high] = a[j]; a[j]=t;
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
      
    public static void dualPivotQuick_sort(int... a) { dualPivotQuick_sort(a, 0, a.length - 1); }
    public static void dualPivotQuick_sort(int[] a, int low, int high) {
        if(low >= high) return;
        int idx, t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high);
        t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        idx = ran.nextInt(low,high);
        t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        dualPivotQuick_sort(a, low, p0 - 1);
        dualPivotQuick_sort(a, p0 + 1, p1 - 1);
        dualPivotQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="merge_sort">
    public static void merge(int[] a, int low, int mid, int high)  {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { insert_sort(a, low, high); return; }
        
        int[] b = new int[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) a[end--] = (a[i] > b[j] ? a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    public static void merge_sort(int... a) { merge_sort(a, 0, a.length - 1); }
    public static void merge_sort(int[] a, int low, int high) {
        if(low >= high) return;
        int mid = (low + high) >> 1;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    static void tim_merge(int[] edge, int len, int[] a) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y withs smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(a, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static boolean tim_sort(int[] a) { return tim_sort(a, 0, a.length - 1); }
    public static boolean tim_sort(int[] a, int low, int high) {
        int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i + 1]]
        for(int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if(a[k] <= a[k + 1]) while((++k < high) && (a[k] <= a[k + 1]));
            else {
                while((++k < high) && (a[k] >= a[k + 1]));
                for(int start = edge[index],end = k; start < end; start++, end--) {
                    int t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, a);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="counting_sort">
    public static boolean counting_sort(int... a) { return counting_sort(a, 0, a.length - 1); }
    public static boolean counting_sort(int[] a, int low, int high) {
        int threshold = COUNTING_SORT_RANGE_INT;
        int free_memory = (int) (Runtime.getRuntime().freeMemory() >> 1);
        if(threshold > free_memory) threshold = free_memory;
        
        MaxMin<Integer> mm = Vector.maxMin(a, low, high, threshold);
        if(mm == null) return false;
        
        int min = mm.min(), dif = mm.max() - min + 1;
        if(dif > threshold) return false;
        
        int[] h = new int[dif];
        for(int i = low; i <= high; i++) h[a[i] - min]++;
        for(int i = 0, index = low; i < dif; i++)
        for(int j = 0; j < h[i]; j++) a[index++] = min + i;
        return true;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thread-sort">
    static void msort(ArrayList<Future<?>> fts, int[] a, int low, int high) {
        int len = high - low + 1, idx, t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, low, p0 - 1); return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); } 
            return;
        }

        msort(fts, a, low, p0 - 1);
        msort(fts, a, p0 + 1, p1 - 1);
        msort(fts, a, p1 + 1, high);
    }
    //</editor-fold>
    public static void sort(int... a) { sort(a, 0, a.length - 1); }
    public static void sort(int[] a, int low, int high) {
        int len = high - low + 1, idx, t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom ran = Lang.exr();
            idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
            idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
            long p = dualPivot_part(a, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        if (tim_sort(a, low, high)) return;
        if (counting_sort(a, low, high)) return;
        if (len > MULTI_THREAD_LEAF) { 
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, a, low, high);
            sync(fts); return; 
        }
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, low, p0 - 1);
        sort(a, p0 + 1, p1 - 1);
        sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sort: long">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static void insert_sort(long[] a, int low) { insert_sort(a, 0, a.length - 1); }
    public static void insert_sort(long[] a, int low, int high) {
        for (int i = low + 1; i <= high; i++) {
            int j; long t;//a[i] < a[j]
            for (t = a[low], a[low] = a[i], j = i - 1; a[low] < a[j]; j--) a[j + 1] = a[j];
            if (j != low) { a[j + 1] = a[low]; a[low] = t; continue; }
            if (a[low] < t) a[low + 1] = t;//k == a[0]
            else { a[low + 1] = a[low]; a[low] = t; }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static void select_sort(long... a) { select_sort(a, 0, a.length - 1); }
    public static void select_sort(long[] a, int low, int high) {
        for (int i = low; i < high; i++)
        for (int j = i+1; j <= high; j++)
            if(a[j] < a[i]) { long t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static void shell_sort(long... a) { shell_sort(a, 0, a.length - 1); }
    public static void shell_sort(long[] a, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; h = h * 3 + 1);
        for (; h > 0; h /= 3) {
            for (int i = h + low; i <= high; i++)
            for (int j = i, p; (j >= h) && (a[j] < a[p = j - h]); j = p) {
                long t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static long dualPivot_part(long[] a, int low ,int high) {
        long t; if(a[low] > a[high]) { t = a[low]; a[low] = a[high]; a[high] = t;}
        
        long p1 = a[low], p2 = a[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if(a[k ] < p1) { t = a[i]; a[i] = a[k]; a[k] = t; i++;}
                else if(a[k] >= p2) break;
                k++;
            }
            while (k <= j) {
                if (a[j] > p2) j--;
                else if (a[j] >= p1&&a[j] <= p2)  {
                    t = a[j]; a[j] = a[k]; a[k] = t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i];a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        t = a[low];  a[low]  = a[i]; a[i]=t;
        t = a[high]; a[high] = a[j]; a[j]=t;
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="merge_sort">
    public static void merge(long[] a, int low, int mid, int high)  {
        if ((high - low + 1) <= INSERT_SORT_THRESHOLD) { insert_sort(a, low, high); return; }
        
        long[] b = new long[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while (i >= low && j >= 0) a[end--] = (a[i] > b[j] ? a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    public static void merge_sort(long... a) { merge_sort(a, 0, a.length - 1); }
    public static void merge_sort(long[] a, int low, int high) {
        if(low >= high) return;
        int mid = (low + high) >> 1;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    static void tim_merge(int[] edge, int len, long[] a) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y withs smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if((st[3] - st[2]) < (st[1] - st[0])) merge(a, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static boolean tim_sort(long[] a) { return tim_sort(a, 0, a.length - 1); }
    public static boolean tim_sort(long[] a, int low, int high) {
        int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i + 1]]
        for (int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if (a[k] <= a[k + 1]) while((++k < high) && (a[k] <= a[k + 1]));
            else {
                while((++k < high) && (a[k] >= a[k + 1]));
                for(int start = edge[index],end = k; start < end; start++, end--) {
                    long t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, a);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thread-sort">
    static void msort(ArrayList<Future<?>> fts, long[] a, int low, int high) {
        int len = high - low + 1, idx; long t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, low, p0 - 1); return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); } 
            return;
        }

        msort(fts, a, low, p0 - 1);
        msort(fts, a, p0 + 1, p1 - 1);
        msort(fts, a, p1 + 1, high);
    }
    //</editor-fold>
    public static void sort(long... a) { sort(a, 0, a.length - 1); }
    public static void sort(long[] a, int low, int high) {
        int len = high - low + 1, idx; long t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom ran = Lang.exr();
            idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
            idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
            long p = dualPivot_part(a, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        if (tim_sort(a, low, high)) return;
        if (len > MULTI_THREAD_LEAF) { 
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, a, low, high);
            sync(fts); return; 
        }
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, low, p0 - 1);
        sort(a, p0 + 1, p1 - 1);
        sort(a, p1 + 1, high);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sort: float">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static void insert_sort(float[] a) { insert_sort(a, 0, a.length - 1); }
    public static void insert_sort(float[] a, int low, int high) {
        for(int i = low + 1; i <= high; i++) {
            int j; float k;//a[i] < a[j]
            for(k = a[low], a[low] = a[i], j = i - 1; a[low] < a[j]; j--) a[j + 1] = a[j];
            if(j != low) { a[j + 1] = a[low]; a[low] = k; continue; }
            if(a[low] < k) a[low + 1] = k;//k == a[0]
            else { a[low + 1] = a[low]; a[low] = k;}
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static void select_sort(float... a) { select_sort(a, 0, a.length - 1); }
    public static void select_sort(float[] a, int low, int high) {
        for(int i = low; i < high; i++)
        for(int j = i+1; j <= high; j++)
            if(a[j ] < a[i]) { float p = a[i]; a[i] = a[j]; a[j] = p;}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static void shell_sort(float... a) { shell_sort(a, 0, a.length - 1); }
    public static void shell_sort(float[] a, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; ) h = h * 3 + 1;
        for(; h > 0; h /= 3) {
            for(int i = h + low; i <= high; i++)
            for(int j = i, p; (j >= h) && (a[j] < a[p = j - h]); j = p) {
                float t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="binaryHeap_sort">
    public static void heap_sort(float... a) { heap_sort(a, 0, a.length - 1); }
    public static void heap_sort(float[] a, int low ,int high) {
        for(int i = (low + high) >> 1; i >= low;i--)//create max-heap
        for(int p = i, left = (p << 1) + 1; left <= high; left = (p << 1) + 1) {
            int max = (a[p] >= a[left] ? p : left);
            if(++left <= high) max = (a[max] >= a[left]? max  :left);
            if(max == p) break;
            float t = a[max]; a[max] = a[p]; a[p] = t;
            p = max;
        }
        
        for(int i = high; i > low;) {//exchange A[0] and A[i], then max-down().
            float t = a[low]; a[low] = a[i]; a[i] = t;
            --i;
            for(int p = 0, left = 1; left <= i; left = (p << 1) + 1) {
                int max = (a[p] >= a[left] ? p : left);
                if(++left <= i) max = (a[max] >= a[left] ? max : left);
                if(max == p) break;
                t = a[max]; a[max] = a[p]; a[p] = t;
                p = max;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="quick_sort">
    public static int quick_part(float[] a, int low, int high) {
        float t, p = a[low];
        while (low < high) {
            while (low < high && a[high] >= p) high--;
            t = a[low]; a[low] = a[high]; a[high] = t;
            while (low < high && a[low] <= p) low++;
            t = a[low]; a[low] = a[high]; a[high] = t;
        }
        return low;
    }
     
    public static void quick_sort(float... a) { quick_sort(a, 0, a.length - 1); }
    public static void quick_sort(float[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low, high);
        float t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        int p = Sort.quick_part(a, low, high);
        quick_sort(a, low, p - 1);
        quick_sort(a, p + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="threeQuick_sort">
    public static long three_part(float[] a, int low, int high) {
        float p = a[low];
        for(int k = low + 1 ;k <= high;) {
            while(k <= high) {
                if(a[k] < p) { float t = a[low]; a[low] = a[k]; a[k] = t; low++;}
                else if(a[k] > p) break;
                k++;
            }
            while(k <= high) {
                if(a[high] > p) high--;
                else if(a[high] == p)  {
                    float t = a[k]; a[k] = a[high]; a[high] = t;
                    k++; high--; break;
                }
                else {
                    float t = a[low];a[low]=a[high];a[high]=a[k];a[k]=t;
                    low++; k++; high--; break;
                }
            }
        }
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
     
    public static void threeQuick_sort(float... a) { threeQuick_sort(a, 0, a.length - 1); }
    public static void threeQuick_sort(float[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low, high);
        float t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        long p = three_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        threeQuick_sort(a, low, p0 - 1);
        threeQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static long dualPivot_part(float[] a, int low ,int high) {
        float t; if(a[low] > a[high]) { t = a[low]; a[low] = a[high]; a[high] = t;}
        
        float p1 = a[low], p2 = a[high];
        int i = low + 1, k = low + 1,j = high - 1;
        while(k <= j) {
            while(k <= j) {
                if(a[k] < p1) { t = a[i]; a[i] = a[k]; a[k] = t; i++; }
                else if(a[k] >= p2) break;
                k++;
            }
            while(k <=j ) {
                if(a[j] > p2) j--;
                else if(a[j] >= p1 && a[j] <= p2) {
                    t = a[j]; a[j] = a[k]; a[k] = t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i]; a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        t = a[low]; a[low] = a[i]; a[i] = t;
        t = a[high]; a[high] = a[j]; a[j] = t;
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
     
    public static void dualPivotQuick_sort(float... a) { dualPivotQuick_sort(a, 0, a.length - 1); }
    public static void dualPivotQuick_sort(float[] a, int low, int high) {
        if(low >= high) return;
        int idx; float t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high);
        t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        idx = ran.nextInt(low,high);
        t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        dualPivotQuick_sort(a, low, p0 - 1);
        dualPivotQuick_sort(a, p0 + 1, p1 - 1);
        dualPivotQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="merge_sort">
    public static void merge(float[] a, int low, int mid, int high)  {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { insert_sort(a, low, high); return; }
        
        float[] b = new float[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) a[end--] = (a[i] > b[j] ? a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    public static void merge_sort(float... a) { merge_sort(a, 0, a.length - 1); }
    public static void merge_sort(float[] a, int low, int high) {
        if(low >= high) return;
        int mid = (low + high) >> 1;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    public static void tim_merge(int[] edge, int len, float[] a) {//len>=1
        int[] st = new int[4];//if there are three segments in stack, then combine y with smaller(z,x)
        for(int  i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(a, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, st[0], st[1]-1, st[2]-1);
    }
    
    public static boolean tim_sort(float... a) { return tim_sort(a, 0, a.length - 1); }
    public static boolean tim_sort(float[] a, int low, int high) {
        float t; int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i+1]]
        for(int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if(a[k] <= a[k + 1]) while((++k < high) && (a[k] <= a[k + 1]));
            else {
                while((++k < high) && (a[k] >= a[k + 1]));
                for(int start = edge[index],end = k; start < end; start++, end--) {
                    t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM) { 
            edge[++index] = high + 1; 
            tim_merge(edge, index, a); 
            return true; 
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi_thread_sort">
    static void msort(ArrayList<Future<?>> fts ,float[] a, int low, int high) {
        int idx; float t;
        
        ExRandom ran = Lang.exr();//random shuffle
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        
        int len = high - low + 1;
        if(len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        
        if(len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, low, p0 - 1);  return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); }
            return;
        }
      
        msort(fts, a, low, p0 - 1);
        msort(fts, a, p0 + 1, p1 - 1);
        msort(fts, a, p1 + 1, high);
    }
    //</editor-fold>
    public static void sort(float... a) { sort(a, 0, a.length - 1); }
    public static void sort(float[] a, int low, int high) {
        int len = high - low + 1, idx; float t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, low, high); return; }//checked
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, low, high); return; }//checked
        if (len <= QUICK_SORT_THRESHOLD) 
        if (tim_sort(a, low, high)) return;
        if (len > MULTI_THREAD_LEAF) {
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, a, low, high); 
            sync(fts); return; 
        }
        
        ExRandom exr = Lang.exr();
        idx = exr.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = exr.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t; 
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, low, p0 - 1);
        sort(a, p0 + 1, p1 - 1);
        sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sort: double">
    //<editor-fold defaultstate="collapsed" desc="insert_sort">
    public static void insert_sort(double... a) { insert_sort(a, 0, a.length - 1); }
    public static void insert_sort(double[] a, int low, int high) {
        for(int i = low + 1; i <= high; i++) {
            int j; double k;//a[i] < a[j]
            for(k = a[low], a[low] = a[i], j = i - 1; a[low] < a[j]; j--) a[j + 1] = a[j];
            if(j != low) { a[j + 1] = a[low]; a[low] = k; continue; }
            if(a[low] < k) a[low + 1] = k;//k == a[0]
            else { a[low + 1] = a[low]; a[low] = k; }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static void select_sort(double... a) { select_sort(a, 0, a.length - 1); }
    public static void select_sort(double[] a, int low, int high) {
        for(int i = low; i < high; i++)
        for(int j = i+1; j <= high; j++)
            if(a[j] < a[i]) { double t = a[i]; a[i] = a[j]; a[j] = t;}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static void shell_sort(double... a) { shell_sort(a, 0, a.length - 1); }
    public static void shell_sort(double[] a, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; h = h*3 + 1);
        for(; h > 0; h /= 3) {
            for(int i = h + low; i <= high; i++)
            for(int j = i, p; (j >= h) && (a[j] < a[p = j - h]); j = p) { 
                double t = a[j]; a[j] = a[p]; a[p] = t;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="binaryHeap_sort">
    public static void heap_sort(double... a) { heap_sort(a, 0, a.length - 1); }
    public static void heap_sort(double[] a, int low ,int high) {
        for(int i = (low + high) >> 1; i >= low; i--)//create max-heap
        for(int p = i, left = (p <<1 ) + 1; left <= high; left = (p << 1) + 1) {
            int max = (a[p] >= a[left]? p : left);
            if(++left <= high) max = (a[max] >= a[left]? max : left);
            if(max == p) break;
            double t = a[max]; a[max] = a[p]; a[p] = t;
            p = max;
        }
        
        for(int i = high; i > low;) {//exchange A[0] and A[i], then max-down().
            double t = a[low]; a[low] = a[i]; a[i] = t;
            --i;
            for(int p = 0, left = 1; left <= i; left = (p << 1) + 1) {
                int max = (a[p] >= a[left] ? p : left);
                if(++left <= i) max = (a[max] >= a[left] ? max : left);
                if(max == p) break;
                t = a[max]; a[max] = a[p]; a[p] = t;
                p = max;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="quick_sort">
    public static int quick_part(double[] a, int low, int high) {
        double t, p = a[low];
        while (low < high) {
            while (low < high && a[high] >= p) high--;
            t = a[low]; a[low] = a[high]; a[high] = t;
            while (low < high && a[low] <= p) low++;
            t = a[low]; a[low] = a[high]; a[high] = t;
        }
        return low;
    }

    public static void quick_sort(double... a) { quick_sort(a, 0, a.length - 1); }
    public static void quick_sort(double[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low,high);
        double t = a[low]; a[low] = a[idx]; a[idx]=t;
        
        int p = quick_part(a, low, high);
        quick_sort(a, low, p - 1);
        quick_sort(a, p + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="threeQuick_sort">
    public static long three_part(double[] a, int low, int high) {
        double p = a[low];
        for(int k = low + 1; k <= high; ) {
            while(k <= high) {
                if(a[k] < p) { double t = a[low]; a[low] = a[k]; a[k] = t; low++;}
                else if(a[k] > p) break;
                k++;
            }
            while(k <= high) {
                if(a[high] > p) high--;
                else if(a[high] == p)  {
                    double t = a[k]; a[k] = a[high]; a[high] = t;
                    k++; high--; break;
                }
                else {
                    double t = a[low]; a[low] = a[high]; a[high] = a[k]; a[k] = t;
                    low++; k++; high--; break;
                }
            }
        }
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
    
    public static void threeQuick_sort(double... a) { threeQuick_sort(a, 0, a.length - 1); }
    public static void threeQuick_sort(double[] a, int low, int high) {
        if(low >= high) return;
        int idx = Lang.exr().nextInt(low, high);
        double t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        long p = three_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        threeQuick_sort(a, low, p0 - 1);
        threeQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static long dualPivot_part(double[] a, int low ,int high) {
        double t; if(a[low] > a[high]) { t = a[low]; a[low] = a[high]; a[high] = t;}
        
        double p1 = a[low], p2 = a[high];
        int i = low + 1, k=low+1, j=high-1;
        while(k <= j) {
            while(k <= j) {
                if(a[k]<p1) { t = a[i]; a[i] = a[k]; a[k] = t; i++; }
                else if(a[k] >= p2) break;
                k++;
            }
            while(k <= j) {
                if(a[j] > p2) j--;
                else if(a[j] >= p1 && a[j] <= p2)  {
                    t = a[j]; a[j] = a[k];a[k]=t;
                    k++; j--; break;
                }
                else {
                    t = a[j]; a[j] = a[k]; a[k] = a[i]; a[i] = t;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        t = a[low]; a[low] = a[i]; a[i] = t;
        t = a[high]; a[high] = a[j]; a[j] = t;
        return ((long)i & 0xFFFFFFFFl) | (((long)j << 32) & 0xFFFFFFFF00000000l);
    }
       
    public static void dualPivotQuick_sort(double... a) { dualPivotQuick_sort(a, 0, a.length - 1); }
    public static void dualPivotQuick_sort(double[] a, int low, int high) {
        if(low >= high) return;
        int idx; double t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high);
        t = a[low]; a[low] = a[idx]; a[idx] = t;
        
        idx = ran.nextInt(low,high);
        t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = Num.low32(p), p1 = Num.high32(p);
        dualPivotQuick_sort(a, low, p0 - 1);
        dualPivotQuick_sort(a, p0 + 1, p1 - 1);
        dualPivotQuick_sort(a, p1 + 1, high);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="merge_sort">
    public static void merge(double[] a, int low, int mid, int high)  {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { insert_sort(a, low, high); return; }
        
        double[] b = new double[high - mid];//from mid+1 to high
        System.arraycopy(a, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) a[end--] = (a[i] > b[j]?  a[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, a, end - j, j + 1);
    }
    
    public static void merge_sort(double... a) { merge_sort(a, 0, a.length - 1); }
    public static void merge_sort(double[] a, int low, int high) {
        if(low >= high) return;
        int mid = (low + high) >> 1;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    public static void tim_merge(int[] edge, int len, double[] a) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y with smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(a, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(a, st[0], st[1] - 1, st[2] - 1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(a, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static boolean tim_sort(double... a) { return tim_sort(a, 0, a.length - 1); }
    public static boolean tim_sort(double[] a, int low, int high) {
        double t; int index = 0;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1]; edge[index] = low;
        
        //find element-ordered segment:each[edge[i], edge[i + 1]]
        for(int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if(a[k] <= a[k + 1]) while((++k < high) && (a[k] <= a[k + 1]));
            else {
                while((++k < high) && (a[k] >= a[k + 1]));
                for(int start = edge[index], end = k; start < end; start++, end--) {
                    t = a[start]; a[start] = a[end]; a[end] = t;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM) {
            edge[++index] = high + 1;
            tim_merge(edge, index, a);
            return true;
        }
        return false;
    }
     //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thread-sort">
    static void msort(ArrayList<Future<?>> fts, double[] a, int low, int high)  {
        int len = high - low + 1, idx; double t;
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0  =(int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(a, low, p0 - 1); return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(a, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(a, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); }
            return;
        }

        msort(fts, a, low, p0 - 1);
        msort(fts, a, p0 + 1, p1 - 1);
        msort(fts, a, p1 + 1, high);
    }
    //</editor-fold>
    public static void sort(double... a) { sort(a, 0, a.length - 1); }
    public static void sort(double[] a, int low, int high) {
        int len = high - low + 1, idx; double t;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(a, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(a, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom ran = Lang.exr();
            idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
            idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
            long p = dualPivot_part(a, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p>>32);
            sort(a, low, p0 - 1);
            sort(a, p0 + 1, p1 - 1);
            sort(a, p1 + 1, high);
            return;
        }
        if(tim_sort(a, low, high)) return;
        if(len > MULTI_THREAD_LEAF) { 
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, a, low, high); 
            sync(fts); return; 
        }
        
        ExRandom ran = Lang.exr();
        idx = ran.nextInt(low, high); t = a[low];  a[low]  = a[idx]; a[idx] = t;
        idx = ran.nextInt(low, high); t = a[high]; a[high] = a[idx]; a[idx] = t;
        
        long p = dualPivot_part(a, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(a, low, p0 - 1);
        sort(a, p0 + 1, p1 - 1);
        sort(a, p1 + 1, high);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sort: key<char>, value">
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static <T> void select_sort(char[] key, T[] val) { select_sort(key, val, 0, key.length - 1); }
    public static <T> void select_sort(char[] key, T[] val, int low, int high) {
        char tk; T tv;
        for (int i = low; i < high; i++)
        for (int j = i + 1; j <= high; j++)
            if(key[j] < key[i]) {
                tk = key[i]; key[i] = key[j]; key[j] = tk;
                tv = val[i]; val[i] = val[j]; val[j] = tv;
            }
    }
    //</editor-fold> 
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static <T> void shell_sort(char[] key, T[] val) { shell_sort(key, val, 0, key.length - 1); }
    public static <T> void shell_sort(char[] key, T[] val, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; ) h = h * 3 + 1;
        char tk; T tv;
        for (; h > 0; h /= 3) {
            for (int i = h + low; i <= high; i++)
            for (int j = i, p; j >= h && key[j] < key[p = j - h]; j = p)  {
                tk = key[j]; key[j] = key[p]; key[p] = tk;
                tv = val[j]; val[j] = val[p]; val[p] = tv;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static <T> long dualPivot_part(char[] key, T[] val, int low ,int high) {
        char tk; T tv; 
        if(key[low] > key[high]) { 
            tk = key[low]; key[low] = key[high]; key[high] = tk;
            tv = val[low]; val[low] = val[high]; val[high] = tv;
        }
        
        int p1 = key[low], p2 = key[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if (key[k] < p1) { 
                    tk = key[i]; key[i] = key[k]; key[k] = tk; 
                    tv = val[i]; val[i] = val[k]; val[k] = tv; 
                    i++;
                }
                else if(key[k] >= p2) break;
                k++;
            }
            while (k <= j) {
                if (key[j] > p2) j--;
                else if (key[j] >= p1 && key[j] <= p2) {
                    tk = key[j]; key[j] = key[k]; key[k] = tk;
                    tv = val[j]; val[j] = val[k]; val[k] = tv;
                    k++; j--; break;
                }
                else {
                    tk = key[j]; key[j] = key[k]; key[k] = key[i]; key[i] = tk;
                    tv = val[j]; val[j] = val[k]; val[k] = val[i]; val[i] = tv;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        
        tk = key[low]; key[low] = key[i]; key[i] = tk;
        tv = val[low]; val[low] = val[i]; val[i] = tv;
        
        tk = key[high]; key[high] = key[j]; key[j]=tk;
        tv = val[high]; val[high] = val[j]; val[j] = tv;
        
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    public static <T> void merge(char[] key, T[] val, int low, int mid, int high)  {
        if (high - low + 1 <= INSERT_SORT_THRESHOLD) { select_sort(key, val, low, high); return; }
        
        char[] b = new char[high - mid]; 
        System.arraycopy(key, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) key[end--] = (key[i] > b[j] ?  key[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, key, end - j, j + 1);
    }
    
    public static <T> void tim_merge(int[] edge, int len, char[] key, T[] val) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y with smaller(z,x)
        for (int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(key, val, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(key, val, st[0], st[1]-1, st[2]-1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(key, val, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static <T> boolean tim_sort(char[] key, T[] val, int low, int high) {
        char tk; T tv;
        int index = 0, start, end;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1];
        edge[index] = low; //find element-ordered segment:each[edge[i], edge[i + 1]]
        
        for (int k= low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if(key[k] <= key[k + 1]) while((++k < high) && (key[k] <= key[k + 1]));
            else {
                while ((++k < high) && key[k] >= key[k + 1]);
                for (start = edge[index],end = k; start < end; start++, end--) {
                    tk = key[start]; key[start] = key[end]; key[end] = tk;
                    tv = val[start]; val[start] = val[end]; val[end] = tv;
                }
            }
        }
        
        if (index == 0) return true;
        if (index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, key, val);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thead-sort">
    static <T> void msort(ArrayList<Future<?>> fts, char[] key, T[] val, int low, int high) {
        int idx; char tk; T tv; ExRandom ran = Lang.exr();//random shuffle
        
        idx = ran.nextInt(low, high); 
        tk = key[low]; key[low] = key[idx]; key[idx] = tk;
        tv = val[low]; val[low] = val[idx]; val[idx] = tv;
        
        idx = ran.nextInt(low, high); 
        tk = key[high]; key[high] = key[idx]; key[idx] = tk;
        tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
        long p = dualPivot_part(key, val, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1=(int) (p >> 32);
        
        int len = high - low + 1;
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(key, low, p0 - 1);
            sort(key, p0 + 1, p1 - 1);
            sort(key, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(key, val, low, p0 - 1);  return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(key, val, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(key, val, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); }
            return;
        }
      
        msort(fts, key, val, low, p0 - 1);
        msort(fts, key, val, p0 + 1, p1 - 1);
        msort(fts, key, val, p1 + 1, high);
    }
    //</editor-fold>
    public static <T> void sort(char[] key, T[] val) { sort(key, val, 0, key.length - 1); }
    public static <T> void sort(char[] key, T[] val, int low, int high) {
        int len = high - low + 1, idx; char tk; T tv;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(key, val, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(key, val, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) { 
            ExRandom exr = Lang.exr();
            idx = exr.nextInt(low, high);
            tk = key[low]; key[low] = key[idx]; key[idx] = tk;
            tv = val[low]; val[low] = val[idx]; val[idx] = tv;
            
            idx = exr.nextInt(low,high);
            tk = key[high]; key[high] = key[idx]; key[idx] = tk;
            tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
            long p = dualPivot_part(key, val, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(key, val, low, p0 - 1);
            sort(key, val, p0 + 1, p1 - 1);
            sort(key, val, p1 + 1, high);
            return;
        }
        if (tim_sort(key, val, low, high)) return;
        if (len > MULTI_THREAD_LEAF) {
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, key, val, low, high); 
            sync(fts); return; 
        }
        
        ExRandom exr = Lang.exr();
        idx = exr.nextInt(low, high);
        tk = key[low]; key[low] = key[idx]; key[idx] = tk;
        tv = val[low]; val[low] = val[idx]; val[idx] = tv;
        
        idx = exr.nextInt(low,high);
        tk = key[high]; key[high] = key[idx]; key[idx] = tk;
        tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
        long p = dualPivot_part(key, val, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(key, val, low, p0 - 1);
        sort(key, val, p0 + 1, p1 - 1);
        sort(key, val, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sort: key<int>, value">
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static <T> void select_sort(int[] key, T[] val) { select_sort(key, val, 0, key.length - 1); }
    public static <T> void select_sort(int[] key, T[] val, int low, int high) {
        int tk; T tv;
        for(int i = low; i < high; i++)
        for(int j = i + 1; j <= high; j++)
            if(key[j] < key[i]) {
                tk = key[i]; key[i] = key[j]; key[j] = tk;
                tv = val[i]; val[i] = val[j]; val[j] = tv;
            }
    }
    //</editor-fold> 
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static <T> void shell_sort(int[] key, T[] val) { shell_sort(key, val, 0, key.length - 1); }
    public static <T> void shell_sort(int[] key, T[] val, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; ) h = h * 3 + 1;
        int tk; T tv;
        for(; h > 0; h /= 3) {
            for(int i = h + low; i <= high; i++)
            for(int j = i, p; j >= h && key[j] < key[p = j - h]; j = p)  {
                tk = key[j]; key[j] = key[p]; key[p] = tk;
                tv = val[j]; val[j] = val[p]; val[p] = tv;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static <T> long dualPivot_part(int[] key, T[] val, int low ,int high) {
        int tk; T tv; 
        if(key[low] > key[high]) { 
            tk = key[low]; key[low] = key[high]; key[high] = tk;
            tv = val[low]; val[low] = val[high]; val[high] = tv;
        }
        
        int p1 = key[low], p2 = key[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while(k <= j) {
            while(k <= j) {
                if(key[k] < p1) { 
                    tk = key[i]; key[i] = key[k]; key[k] = tk; 
                    tv = val[i]; val[i] = val[k]; val[k] = tv; 
                    i++;
                }
                else if(key[k] >= p2) break;
                k++;
            }
            while(k <= j) {
                if(key[j] > p2) j--;
                else if(key[j] >= p1 && key[j] <= p2) {
                    tk = key[j]; key[j] = key[k]; key[k] = tk;
                    tv = val[j]; val[j] = val[k]; val[k] = tv;
                    k++; j--; break;
                }
                else {
                    tk = key[j]; key[j] = key[k]; key[k] = key[i]; key[i] = tk;
                    tv = val[j]; val[j] = val[k]; val[k] = val[i]; val[i] = tv;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        
        tk = key[low]; key[low] = key[i]; key[i] = tk;
        tv = val[low]; val[low] = val[i]; val[i] = tv;
        
        tk = key[high]; key[high] = key[j]; key[j]=tk;
        tv = val[high]; val[high] = val[j]; val[j] = tv;
        
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    public static <T> void merge(int[] key, T[] val, int low, int mid, int high)  {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { select_sort(key, val, low, high); return; }
        
        int[] b = new int[high - mid]; 
        System.arraycopy(key, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) key[end--] = (key[i] > b[j] ?  key[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, key, end - j, j + 1);
    }
    
    public static <T> void tim_merge(int[] edge, int len, int[] key, T[] val) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y with smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(key, val, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(key, val, st[0], st[1]-1, st[2]-1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(key, val, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static <T> boolean tim_sort(int[] key, T[] val, int low, int high) {
        int tk; T tv;
        int index = 0, start, end;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1];
        edge[index] = low; //find element-ordered segment:each[edge[i], edge[i + 1]]
        
        for(int k= low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if(key[k] <= key[k + 1]) while((++k < high) && (key[k] <= key[k + 1]));
            else {
                while((++k < high) && key[k] >= key[k + 1]);
                for(start = edge[index],end = k; start < end; start++, end--) {
                    tk = key[start]; key[start] = key[end]; key[end] = tk;
                    tv = val[start]; val[start] = val[end]; val[end] = tv;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, key, val);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thead-sort">
    static <T> void msort(ArrayList<Future<?>> fts, int[] key, T[] val, int low, int high) {
        int idx; int tk; T tv; ExRandom ran = Lang.exr();//random shuffle
        
        idx = ran.nextInt(low, high); 
        tk = key[low]; key[low] = key[idx]; key[idx] = tk;
        tv = val[low]; val[low] = val[idx]; val[idx] = tv;
        
        idx = ran.nextInt(low, high); 
        tk = key[high]; key[high] = key[idx]; key[idx] = tk;
        tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
        long p = dualPivot_part(key, val, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1=(int) (p >> 32);
        
        int len = high - low + 1;
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(key, low, p0 - 1);
            sort(key, p0 + 1, p1 - 1);
            sort(key, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(key, val, low, p0 - 1);  return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(key, val, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(key, val, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); }
            return;
        }
      
        msort(fts, key, val, low, p0 - 1);
        msort(fts, key, val, p0 + 1, p1 - 1);
        msort(fts, key, val, p1 + 1, high);
    }
    //</editor-fold>
    public static <T> void sort(int[] key, T[] val) { sort(key, val, 0, key.length - 1); }
    public static <T> void sort(int[] key, T[] val, int low, int high) {
        int len = high - low + 1, idx; int tk; T tv;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(key, val, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(key, val, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom exr = Lang.exr();
            idx = exr.nextInt(low, high);
            tk = key[low]; key[low] = key[idx]; key[idx] = tk;
            tv = val[low]; val[low] = val[idx]; val[idx] = tv;
            
            idx = exr.nextInt(low,high);
            tk = key[high]; key[high] = key[idx]; key[idx] = tk;
            tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
            long p = dualPivot_part(key, val, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(key, val, low, p0 - 1);
            sort(key, val, p0 + 1, p1 - 1);
            sort(key, val, p1 + 1, high);
            return;
        }
        if(tim_sort(key, val, low, high)) return;
        if(len > MULTI_THREAD_LEAF) {
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, key, val, low, high); 
            sync(fts); return; 
        }
        
        ExRandom exr = Lang.exr();
        idx = exr.nextInt(low, high);
        tk = key[low]; key[low] = key[idx]; key[idx] = tk;
        tv = val[low]; val[low] = val[idx]; val[idx] = tv;
        
        idx = exr.nextInt(low,high);
        tk = key[high]; key[high] = key[idx]; key[idx] = tk;
        tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
        long p = dualPivot_part(key, val, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(key, val, low, p0 - 1);
        sort(key, val, p0 + 1, p1 - 1);
        sort(key, val, p1 + 1, high);
    }
    
    public static <T> void shuffle_sort(T[] val) {
        int size = val.length;
        int[] key = Vector.random_int_vector(size);
        sort(key, val);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sort: key<float>, value">
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static <T> void select_sort(float[] key, T[] val) { select_sort(key, val, 0, key.length - 1); }
    public static <T> void select_sort(float[] key, T[] val, int low, int high) {
        float tk; T tv;
        for (int i = low; i < high; i++)
        for (int j = i + 1; j <= high; j++)
            if(key[j] < key[i]) {
                tk = key[i]; key[i] = key[j]; key[j] = tk;
                tv = val[i]; val[i] = val[j]; val[j] = tv;
            }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static <T> void shell_sort(float[] key, T[] val) { shell_sort(key, val, 0, key.length - 1); }
    public static <T> void shell_sort(float[] key, T[] val, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; ) h = h * 3 + 1;
        float tk; T tv;
        for (; h > 0; h /= 3) {
            for(int i = h + low; i <= high; i++)
            for(int j = i, p; j >= h && key[j] < key[p = j - h]; j = p)  {
                tk = key[j]; key[j] = key[p]; key[p] = tk;
                tv = val[j]; val[j] = val[p]; val[p] = tv;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static <T> long dualPivot_part(float[] key, T[] val, int low ,int high) {
        float tk; T tv; 
        if(key[low] > key[high]) { 
            tk = key[low]; key[low] = key[high]; key[high] = tk;
            tv = val[low]; val[low] = val[high]; val[high] = tv;
        }
        
        float p1 = key[low], p2 = key[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if (key[k] < p1) { 
                    tk = key[i]; key[i] = key[k]; key[k] = tk; 
                    tv = val[i]; val[i] = val[k]; val[k] = tv; 
                    i++;
                }
                else if(key[k] >= p2) break;
                k++;
            }
            while (k <= j) {
                if (key[j] > p2) j--;
                else if (key[j] >= p1 && key[j] <= p2) {
                    tk = key[j]; key[j] = key[k]; key[k] = tk;
                    tv = val[j]; val[j] = val[k]; val[k] = tv;
                    k++; j--; break;
                }
                else {
                    tk = key[j]; key[j] = key[k]; key[k] = key[i]; key[i] = tk;
                    tv = val[j]; val[j] = val[k]; val[k] = val[i]; val[i] = tv;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        
        tk = key[low]; key[low] = key[i]; key[i] = tk;
        tv = val[low]; val[low] = val[i]; val[i] = tv;
        
        tk = key[high]; key[high] = key[j]; key[j]=tk;
        tv = val[high]; val[high] = val[j]; val[j] = tv;
        
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    public static <T> void merge(float[] key, T[] val, int low, int mid, int high)  {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { select_sort(key, val, low, high); return; }
        
        float[] b = new float[high - mid]; 
        System.arraycopy(key, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) key[end--] = (key[i] > b[j] ?  key[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, key, end - j, j + 1);
    }
    
    public static <T> void tim_merge(int[] edge, int len, float[] key, T[] val) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y with smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(key, val, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(key, val, st[0], st[1]-1, st[2]-1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(key, val, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static <T> boolean tim_sort(float[] key, T[] val, int low, int high) {
        float tk; T tv;
        int index = 0, start, end;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1];
        edge[index] = low; //find element-ordered segment:each[edge[i], edge[i + 1]]
        
        for (int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if (key[k] <= key[k + 1]) while((++k < high) && (key[k] <= key[k + 1]));
            else {
                while((++k < high) && key[k] >= key[k + 1]);
                for(start = edge[index],end = k; start < end; start++, end--) {
                    tk = key[start]; key[start] = key[end]; key[end] = tk;
                    tv = val[start]; val[start] = val[end]; val[end] = tv;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, key, val);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thead-sort">
    static <T> void msort(ArrayList<Future<?>> fts, float[] key, T[] val, int low, int high) {
        int idx; float tk; T tv; ExRandom ran = Lang.exr();//random shuffle
        
        idx = ran.nextInt(low, high); 
        tk = key[low]; key[low] = key[idx]; key[idx] = tk;
        tv = val[low]; val[low] = val[idx]; val[idx] = tv;
        
        idx = ran.nextInt(low, high); 
        tk = key[high]; key[high] = key[idx]; key[idx] = tk;
        tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
        long p = dualPivot_part(key, val, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1=(int) (p >> 32);
        
        int len = high - low + 1;
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(key, low, p0 - 1);
            sort(key, p0 + 1, p1 - 1);
            sort(key, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(key, val, low, p0 - 1);  return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(key, val, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(key, val, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); }
            return;
        }
      
        msort(fts, key, val, low, p0 - 1);
        msort(fts, key, val, p0 + 1, p1 - 1);
        msort(fts, key, val, p1 + 1, high);
    }
    //</editor-fold>
    public static <T> void sort(float[] key, T[] val) { sort(key, val, 0, key.length - 1); }
    public static <T> void sort(float[] key, T[] val, int low, int high) {
        int len = high - low + 1, idx; float tk; T tv;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(key, val, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(key, val, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom exr = Lang.exr();
            idx = exr.nextInt(low, high);
            tk = key[low]; key[low] = key[idx]; key[idx] = tk;
            tv = val[low]; val[low] = val[idx]; val[idx] = tv;
            
            idx = exr.nextInt(low,high);
            tk = key[high]; key[high] = key[idx]; key[idx] = tk;
            tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
            long p = dualPivot_part(key, val, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(key, val, low, p0 - 1);
            sort(key, val, p0 + 1, p1 - 1);
            sort(key, val, p1 + 1, high);
            return;
        }
        if (tim_sort(key, val, low, high)) return;
        if (len > MULTI_THREAD_LEAF) {
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, key, val, low, high); 
            sync(fts); return; 
        }
        
        ExRandom exr = Lang.exr();
        idx = exr.nextInt(low, high);
        tk = key[low]; key[low] = key[idx]; key[idx] = tk;
        tv = val[low]; val[low] = val[idx]; val[idx] = tv;
        
        idx = exr.nextInt(low,high);
        tk = key[high]; key[high] = key[idx]; key[idx] = tk;
        tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
        long p = dualPivot_part(key, val, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(key, val, low, p0 - 1);
        sort(key, val, p0 + 1, p1 - 1);
        sort(key, val, p1 + 1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sort: key<double>, value">
    //<editor-fold defaultstate="collapsed" desc="select_sort">
    public static <T> void select_sort(double[] key, T[] val) { select_sort(key, val, 0, key.length - 1); }
    public static <T> void select_sort(double[] key, T[] val, int low, int high) {
        double tk; T tv;
        for (int i = low; i < high; i++)
        for (int j = i + 1; j <= high; j++)
            if(key[j] < key[i]) {
                tk = key[i]; key[i] = key[j]; key[j] = tk;
                tv = val[i]; val[i] = val[j]; val[j] = tv;
            }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="shell_sort">
    public static <T> void shell_sort(double[] key, T[] val) { shell_sort(key, val, 0, key.length - 1); }
    public static <T> void shell_sort(double[] key, T[] val, int low, int high) {
        int h = 1; for(int top = (high - low + 1) / 3; h < top; ) h = h * 3 + 1;
        double tk; T tv;
        for (; h > 0; h /= 3) {
            for (int i = h + low; i <= high; i++)
            for (int j = i, p; j >= h && key[j] < key[p = j - h]; j = p)  {
                tk = key[j]; key[j] = key[p]; key[p] = tk;
                tv = val[j]; val[j] = val[p]; val[p] = tv;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dualPivotQuick_sort">
    public static <T> long dualPivot_part(double[] key, T[] val, int low ,int high) {
        double tk; T tv; 
        if(key[low] > key[high]) { 
            tk = key[low]; key[low] = key[high]; key[high] = tk;
            tv = val[low]; val[low] = val[high]; val[high] = tv;
        }
        
        double p1 = key[low], p2 = key[high];
        int i = low + 1,k = low + 1,j = high - 1;
        while (k <= j) {
            while (k <= j) {
                if (key[k] < p1) { 
                    tk = key[i]; key[i] = key[k]; key[k] = tk; 
                    tv = val[i]; val[i] = val[k]; val[k] = tv; 
                    i++;
                }
                else if(key[k] >= p2) break;
                k++;
            }
            while (k <= j) {
                if (key[j] > p2) j--;
                else if (key[j] >= p1 && key[j] <= p2) {
                    tk = key[j]; key[j] = key[k]; key[k] = tk;
                    tv = val[j]; val[j] = val[k]; val[k] = tv;
                    k++; j--; break;
                }
                else {
                    tk = key[j]; key[j] = key[k]; key[k] = key[i]; key[i] = tk;
                    tv = val[j]; val[j] = val[k]; val[k] = val[i]; val[i] = tv;
                    k++; i++; j--; break;
                }
            }
        }
        i--; j++;
        
        tk = key[low]; key[low] = key[i]; key[i] = tk;
        tv = val[low]; val[low] = val[i]; val[i] = tv;
        
        tk = key[high]; key[high] = key[j]; key[j]=tk;
        tv = val[high]; val[high] = val[j]; val[j] = tv;
        
        return ((long) i & 0xffffffffL) | (((long) j << 32) & 0xffffffff00000000L);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="tim_sort">
    public static <T> void merge(double[] key, T[] val, int low, int mid, int high)  {
        if(high - low + 1 <= INSERT_SORT_THRESHOLD) { select_sort(key, val, low, high); return; }
        
        double[] b = new double[high - mid]; 
        System.arraycopy(key, mid + 1, b, 0, b.length);
        
        int i = mid, j = b.length - 1, end = high;
        while(i >= low && j >= 0) key[end--] = (key[i] > b[j] ?  key[i--] : b[j--]);
        if(j >= 0) System.arraycopy(b, 0, key, end - j, j + 1);
    }
    
    public static <T> void tim_merge(int[] edge, int len, double[] key, T[] val) {//len >= 1
        int[] st = new int[4];//if there are three segments in stack, then combine y with smaller(z,x)
        for(int i = 0, index = 0; i <= len; i++) {
            st[index++] = edge[i];
            if(index == 4) {
                if(st[3] - st[2] < st[1] - st[0]) merge(key, val, st[1], st[2] - 1, st[3] - 1);//merge y and x
                else { merge(key, val, st[0], st[1]-1, st[2]-1); st[1] = st[2]; }//merge y and z
                st[2] = st[3]; index--;
            }
        }
        merge(key, val, st[0], st[1] - 1, st[2] - 1);
    }
    
    public static <T> boolean tim_sort(double[] key, T[] val, int low, int high) {
        double tk; T tv;
        int index = 0, start, end;
        int[] edge = new int[TIM_SORT_SEGMENT_NUM + 1];
        edge[index] = low; //find element-ordered segment:each[edge[i], edge[i + 1]]
        
        for (int k = low; (k < high) && (index < TIM_SORT_SEGMENT_NUM); edge[++index] = ++k) {
            if (key[k] <= key[k + 1]) while ((++k < high) && (key[k] <= key[k + 1]));
            else {
                while ((++k < high) && (key[k] >= key[k + 1]));
                for (start = edge[index], end = k; start < end; start++, end--) {
                    tk = key[start]; key[start] = key[end]; key[end] = tk;
                    tv = val[start]; val[start] = val[end]; val[end] = tv;
                }
            }
        }
        
        if(index == 0) return true;
        if(index < TIM_SORT_SEGMENT_NUM)  {
            edge[++index] = high + 1;
            tim_merge(edge, index, key, val);
            return true;
        }
        return false;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="multi-thead-sort">
    static <T> void msort(ArrayList<Future<?>> fts, double[] key, T[] val, int low, int high) {
        int idx; double tk; T tv;
        
        ExRandom ran = Lang.exr();//random shuffle
        idx = ran.nextInt(low, high); 
        tk = key[low]; key[low] = key[idx]; key[idx] = tk;
        tv = val[low]; val[low] = val[idx]; val[idx] = tv;
        
        idx = ran.nextInt(low, high); 
        tk = key[high]; key[high] = key[idx]; key[idx] = tk;
        tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
        long p = dualPivot_part(key, val, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1=(int) (p >> 32);
        
        int len = high - low + 1;
        if (len <= SINGLE_THREAD_THRESHOLD) {
            sort(key, low, p0 - 1);
            sort(key, p0 + 1, p1 - 1);
            sort(key, p1 + 1, high);
            return;
        }
        
        if (len <= MULTI_THREAD_LEAF) {//the leaf of multi-thread sorting Tree
            Future<?> ft0 = exec.submit(() -> { sort(key, val, low, p0 - 1);  return 0; });
            Future<?> ft1 = exec.submit(() -> { sort(key, val, p0 + 1, p1 - 1); return 0; });
            Future<?> ft2 = exec.submit(() -> { sort(key, val, p1 + 1, high); return 0; });
            synchronized(fts) { fts.add(ft0); fts.add(ft1); fts.add(ft2); }
            return;
        }
      
        msort(fts, key, val, low, p0 - 1);
        msort(fts, key, val, p0 + 1, p1 - 1);
        msort(fts, key, val, p1 + 1, high);
    }
    //</editor-fold>
    public static <T> void sort(double[] key, T[] val) { sort(key, val, 0, key.length - 1); }
    public static <T> void sort(double[] key, T[] val, int low, int high) {
        int len = high - low + 1, idx; double tk; T tv;
        if (len <= 1) return;
        if (len <= SELECT_SORT_THRESHOLD) { select_sort(key, val, low, high); return; }
        if (len <= SHELL_SORT_THRESHOLD) { shell_sort(key, val, low, high); return; }
        if (len <= QUICK_SORT_THRESHOLD) {
            ExRandom exr = Lang.exr();
            idx = exr.nextInt(low, high);
            tk = key[low]; key[low] = key[idx]; key[idx] = tk;
            tv = val[low]; val[low] = val[idx]; val[idx] = tv;
            
            idx = exr.nextInt(low,high);
            tk = key[high]; key[high] = key[idx]; key[idx] = tk;
            tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
            long p = dualPivot_part(key, val, low, high);
            int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
            sort(key, val, low, p0 - 1);
            sort(key, val, p0 + 1, p1 - 1);
            sort(key, val, p1 + 1, high);
            return;
        }
        if (tim_sort(key, val, low, high)) return;
        if (len > MULTI_THREAD_LEAF) {
            ArrayList<Future<?>> fts = new ArrayList<>(4);
            msort(fts, key, val, low, high); 
            sync(fts); return; 
        }
        
        ExRandom exr = Lang.exr();
        idx = exr.nextInt(low, high);
        tk = key[low]; key[low] = key[idx]; key[idx] = tk;
        tv = val[low]; val[low] = val[idx]; val[idx] = tv;
        
        idx = exr.nextInt(low,high);
        tk = key[high]; key[high] = key[idx]; key[idx] = tk;
        tv = val[high]; val[high] = val[idx]; val[idx] = tv;
        
        long p = dualPivot_part(key, val, low, high);
        int p0 = (int) (p & 0x000000ffffffffL), p1 = (int) (p >> 32);
        sort(key, val, low, p0 - 1);
        sort(key, val, p0 + 1, p1 - 1);
        sort(key, val, p1 + 1, high);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Find a element in an Array sorted in asscending order">
    public static int bi_find(int[] a, int val) { return bi_find(a, val, 0, a.length - 1); }
    public static int bi_find(int[] a, int val, int low, int high) {
        while(low <= high) {
            int mid = (low + high) >> 1;
            if (val < a[mid]) high = mid - 1;
            else if (val > a[mid]) low = mid + 1;
            else return mid;
        }
        return -1;
    }
    
    public static int tri_find(int[] a, int val){ return tri_find(a, val, 0, a.length - 1); }
    public static int tri_find(int[] a, int val, int low, int high) {
        while(low <= high) {
            int d = (high - low + 1) / 3;
            int p1 = d + low;
            int p2 = p1 + d;
            
            if(val < a[p1]) high = p1 - 1;
            else if(val > a[p2]) low = p2 + 1;
            else if(val > a[p1] && val < a[p2]) { low = p1 + 1; high = p2 - 1; }
            else if(val == a[p1]) return p1;
            else return p2;
        }
        return -1;
    }
    //</editor-fold>
}
