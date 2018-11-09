import {Component, OnInit} from '@angular/core';
import {IP_ADDRESS} from '../data';
import {Router} from '@angular/router';
import {Http} from '@angular/http';

@Component({
  selector: 'app-menu',
  templateUrl: './menu.component.html',
  styleUrls: ['./menu.component.css']
})
export class MenuComponent implements OnInit {

  selectedDrinksId: number;

  constructor(private http: Http, private router: Router) {
  }

  ngOnInit() {
    this.getCommand();
  }

  getCommand(): void {
    const timerId = setInterval(timer => {
      this.http.get(IP_ADDRESS + '/menuOptions').subscribe(data => {
        console.log(data);
        if (data['_body'] === '1') {
          this.selectedDrinksId = 1;
          this.gotoDetail();
          clearInterval(timerId);
        } else if (data['_body'] === '2') {
          this.selectedDrinksId = 2;
          this.gotoDetail();
          clearInterval(timerId);
        }
      });
    }, 2000);
  }

  gotoDetail(): void {
    this.router.navigate(['/loading', this.selectedDrinksId]);
  }

}
