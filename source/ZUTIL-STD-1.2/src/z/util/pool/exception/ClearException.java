/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool.exception;

/**
 *
 * @author dell
 */
public class ClearException extends RuntimeException
{
    public static final String MSG="PoolClearException: ";
    
    public ClearException()
    {
        super(MSG);
    }
    public ClearException(String msg)
    {
        super(MSG+msg);
    }
    public ClearException(Throwable e)
    {
        super(e);
    }
}
