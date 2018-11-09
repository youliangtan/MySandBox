/* tslint:disable:no-unused-variable */
import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { DebugElement } from '@angular/core';

import { HorribleImageComponent } from './horrible-image.component';

describe('HorribleImageComponent', () => {
  let component: HorribleImageComponent;
  let fixture: ComponentFixture<HorribleImageComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ HorribleImageComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(HorribleImageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
