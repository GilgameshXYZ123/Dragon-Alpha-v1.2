/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.concurrent;

/**
 *
 * @author dell
 */
public class BinarySemaphore
{
    volatile boolean value = false;
    
    public void P() {//get resource
        try { synchronized(this) { while(value) this.wait(); value = true; } }
        catch(InterruptedException e) { }
    }
    
    public void P(long waitTime) {//get resource
        try { synchronized(this) { while(value) this.wait(waitTime); value = true; } }
        catch(InterruptedException e) { }
    }
    
    public synchronized void V() {//realse resource
        if(value) { this.notify(); value = false; }
    }

    public void sleep(long sleepTime) {
        try { synchronized(this) { Thread.sleep(sleepTime); } }
        catch(InterruptedException e) {}
    }
}
