package com.example.fh_2023_tecs.activities;

import static android.widget.Toast.LENGTH_SHORT;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;

import com.example.fh_2023_tecs.R;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.parse.ParseException;
import com.parse.ParseObject;
import com.parse.ParseUser;

import com.example.fh_2023_tecs.fragments.HomeFragment;
import com.example.fh_2023_tecs.fragments.ProfileFragment;
import com.example.fh_2023_tecs.fragments.WikiFragment;
import com.parse.SaveCallback;

import java.util.Arrays;

public class ManualActivity extends AppCompatActivity {

    private BottomNavigationView bottomNavigationView;

    public MyClass(int objectID) {
        this.objectID = objectID; // Here, we use "this" to refer to the instance variable
    }
    public int getObjectID() {
        return this.objectID; // Here, we use "this" to refer to the instance variable
    }

    EditText etItemName;
    EditText etItemType;
    Button btnCheck;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_manual);

        ActionBar actionBar;
        actionBar = getSupportActionBar();

        bottomNavigationView = findViewById(R.id.bottomNavigation);
        etItemName = findViewById(R.id.etItemName);
        etItemType = findViewById(R.id.etItemType);
        btnCheck = findViewById(R.id.btnCheck);

        bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem item) {
                Fragment fragment = null;
                switch (item.getItemId()) {
                    case R.id.action_home:
                        getSupportFragmentManager().beginTransaction()
                                .replace(R.id.flContainer, new HomeFragment()).commit();
                        finish();
                        return true;
                    case R.id.action_wiki:
                        getSupportFragmentManager().beginTransaction()
                                .replace(R.id.flContainer, new WikiFragment()).commit();
                        finish();
                        return true;
                    case R.id.action_profile:
                        getSupportFragmentManager().beginTransaction()
                                .replace(R.id.flContainer, new ProfileFragment(ParseUser.getCurrentUser())).commit();
                        finish();
                        return true;
                    default:
                        return false;
                }
            }
        });

        String itemName = etItemName.getText().toString();
        String itemType = etItemType.getText().toString();
        btnCheck.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                //TODO: fix &&
                if(TextUtils.isEmpty(etItemName.getText().toString()) && TextUtils.isEmpty(etItemType.getText().toString())) {
                    Toast.makeText(ManualActivity.this, "Please fill out all fields!", LENGTH_SHORT).show();
                }
                else {
                    Toast.makeText(ManualActivity.this, "Checking!", LENGTH_SHORT).show();

                    //TODO: DB checking here
                    // Configure Query
                    ParseObject userManualForm = new ParseObject("UserManualForm");
                    // Store an object
                    userManualForm.put("itemName", etItemName.getText().toString());
                    userManualForm.put("itemType", etItemType.getText().toString());
                    this.objectID = etItemName.getText().toString()

                    // Saving object
                    userManualForm.saveInBackground(new SaveCallback() {
                        @Override
                        public void done(ParseException e) {
                            if (e == null) {
                                // Success
                            } else {
                                // Error
                            }
                        }
                    });
                    //TODO: upon success navigate to results, upon error indicate no match/can't be recycled
                    goResultActivity();
                }
            }
        });
    }

    private void goResultActivity() {
        Intent i = new Intent(this, ResultActivity.class);
        startActivity(i);
        finish();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.logout, menu);
        return true;
    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == R.id.Logout) {
            ParseUser.logOut();
            ParseUser currentUser = ParseUser.getCurrentUser();
            goLoginActivity();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }
    private void goLoginActivity() {
        Intent i = new Intent(this, LoginActivity.class);
        startActivity(i);
        finish();
    }
}