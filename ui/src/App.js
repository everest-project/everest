import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import Box from '@material-ui/core/Box';
import Paper from '@material-ui/core/Paper';
import {makeStyles} from '@material-ui/core/styles';
import React from 'react';
import SimpleQueryPanel from './QueryPanel';
import PreviewPanel from './PreviewPanel';
import ResultPanel from './ResultPanel'
import { usePromiseTracker } from "react-promise-tracker";
import Backdrop from '@material-ui/core/Backdrop';
import CircularProgress from '@material-ui/core/CircularProgress';
import SimplePreviewPanel from "./SimplePreviewPanel";

const useStyles = makeStyles((theme) => ({
	root: {
		backgroundColor: theme.palette.background.default,
		display: 'flex',
		flexDirection: 'column',
		height: '100%',
		overflow: 'hidden',
		width: '100%'
	},
	appbar: {
		height: 64
	},
	title: {
		flexGrow: 1
	},
	content: {
		height: "100%",
		backgroundColor: "#F4F6F8",
		display: "flex",
		flexDirection: "row"
	},
	query: {
		flexGrow: 1,
		backgroundColor: "white",
		height: "100%",
		borderRightColor: "rgba(0,0,0,0.12)",
		borderRightWidth: "1px",
		borderRightStyle: "solid",
		boxShadow: "5px 0px 10px 0px rgba(0,0,0,0.20)"
	},
	visualization: {
		flexGrow: 10,
		display: "flex",
		flexDirection: "column",
		height: "100%",
		overflow:"scroll"
	},
	preview: {
		margin: "1em",
		width: "calc(100%-2em)",
	},
	result: {
		margin: "1em",
		marginTop: 0,
		width: "calc(100%-2em)",
		flexGrow: 1
	},

	backdrop: {
    zIndex: 1300,
    color: '#fff',
  },
}));

function LoadingIndicator(props){
	const classes = useStyles();

	const { promiseInProgress } = usePromiseTracker();

	return promiseInProgress &&  (
			<Backdrop className={classes.backdrop} open={true}>
  			<CircularProgress color="inherit" />
			</Backdrop>
	)
}

function App(props) {
	const classes = useStyles();
	const [query, set_query] = React.useState({
		"k": "1",
		"thres": 0.9,
		"window": "1 frame",
		"udf":"",
		"udf_params":{}
	})
	const [video_param, set_video_param] = React.useState({})
	const [result_img_paths, set_result_img_paths] = React.useState([])
	const on_query_change = (new_params) => {
		var new_query = {
			...query,
			...new_params
		}
		set_query(new_query)
		console.log(`new query:`, new_query)
	}

	window.onerror = function (msg, url, line) {
    console.log("Caught[via window.onerror]: '" + msg + "' from " + url + ":" + line);
    return true; // same as preventDefault
	};

	window.addEventListener('error', function (evt) {
			console.log("Caught[via 'error' event]:  '" + evt.message + "' from " + evt.filename + ":" + evt.lineno);
			console.log(evt); // has srcElement / target / etc
			evt.preventDefault();
	});


	return (
		<Box className={classes.root}>
			<AppBar position="static" elevation={5} className={classes.appbar}>
				<Toolbar>
					<Typography className={classes.title} variant="h5">
						Everest
						</Typography>
					<Button className={classes.paper} color="inherit">Paper</Button>
				</Toolbar>
			</AppBar>
			<Box className={classes.content}>
				<Box className={classes.query}>
					<SimpleQueryPanel on_query_change={on_query_change} query={query} on_video_change={set_video_param} set_result_img_paths={set_result_img_paths} video_param={video_param}/>
				</Box>
				<Box className={classes.visualization}>
					{/*<Paper className={classes.preview} elevation={2}>*/}
					{/*	<SimplePreviewPanel query={query} video_param={video_param}/>*/}
					{/*</Paper>*/}
					<Paper className={classes.result} elevation={2}>
						<ResultPanel result_img_paths={result_img_paths}/>
					</Paper>
				</Box>
			</Box>
		<LoadingIndicator />
		</Box>
	);
}

export default App;

