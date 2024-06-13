/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.concurrent;

/**
 *
 * @author Gilgamesh
 */
public class SimpleLock 
{
    private volatile boolean lock = false;
    
    public final synchronized void lock() { lock = true; }
    
    public final synchronized void unlock() { lock = false; this.notifyAll(); }
    
    public final synchronized void require() {  try { if(lock) this.wait(); } catch(InterruptedException e) { } }
}
