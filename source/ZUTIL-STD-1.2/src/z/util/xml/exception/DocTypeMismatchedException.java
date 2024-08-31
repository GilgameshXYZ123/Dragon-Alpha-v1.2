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
public class DocTypeMismatchedException extends RuntimeException
{
    public static final String MESSAGE="DocTypeMismatchedException: ";
    
    public DocTypeMismatchedException() {
        super();
    }
    
    public DocTypeMismatchedException(String message) {
        super(MESSAGE+message);
    }
}
