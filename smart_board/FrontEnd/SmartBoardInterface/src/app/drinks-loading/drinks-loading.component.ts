import {Component, OnInit} from '@angular/core';
import {ActivatedRoute, Router} from '@angular/router';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/switchMap';
import {IP_ADDRESS} from '../data';
import {Http} from '@angular/http';

@Component({
  selector: 'app-drinks-loading',
  templateUrl: './drinks-loading.component.html',
  styleUrls: ['./drinks-loading.component.css']
})
export class DrinksLoadingComponent implements OnInit {

  private selectedDrinkID: number;

  constructor(private activatedRoute: ActivatedRoute,
              private http: Http,
              private router: Router) {
  }

  ngOnInit() {
    this.activatedRoute.params
      .map(params => params['id'])
      .subscribe(id => {
        console.log(id);
        this.selectedDrinkID = id;
        this.getCommand();
      });
  }

  getCommand(): void {
    const timerId = setInterval(timer => {
      this.http.get(IP_ADDRESS + '/dispenseStatus').subscribe(data => {
        console.log(data['_body']);
        if (data['_body'] === 'Next') {
          this.gotoDetail();
          clearInterval(timerId);
        }
      });
    }, 2000);
  }

  gotoDetail(): void {
    this.router.navigate(['/result', this.selectedDrinkID]);
  }


}
