/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.xml.exception;

/**
 *
 * @author dell
 */
public class UnknownTagException extends RuntimeException
{
    public static final String MESSAGE="UnknownTagException";
            
    public UnknownTagException()
    {
        super();
    }
    public UnknownTagException(String message)
    {
        super(MESSAGE+message);
    }
}
