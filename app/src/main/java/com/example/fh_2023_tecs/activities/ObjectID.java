import mypackage.ManualActivity;

public class ObjectID {
    public static String getParameter() {
        ManualActivity myObject = new ManualActivity(42);
        String objectID = myObject.getObjectID();
        return objectID;
    }
}