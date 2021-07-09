import {makeStyles} from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import React, {useCallback, useState} from 'react';
import ResponsiveGallery from 'react-responsive-gallery';
import Grid from "@material-ui/core/Grid";
import { Player } from 'video-react';
import ModalImage from "react-modal-image";

const useStyles = makeStyles((theme) => ({
    root: {
        width: "auto",
        height: "100%",
        margin: "0.5em 1.5em",
    },
    result_title: {
        display: "block",
        color: "#3f51b5"
    },
    gallery:{
        height: "100%",
        width: "100%",
        display: "block",
    },

    result_gallery_wrapper:{
        width: "100%",
        overflow:"auto"
    },
    entry_box: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center"
    }

}));

let convert_imgs_to_gallery_obj = (imgs) => {
    console.log("Converting: ", imgs)
    let ret = []
    for(let idx in imgs){
        let img_src = imgs[idx]
        ret.push({
            "src":img_src,
            // "width": 1,
            // "height": 1
            // "thumbnail":img_src,
            // "thumbnailWidth": 320,
            // "thumbnailHeight": 174,
        })
    }
    return ret
}

function ResultPanel(props) {
    const classes = useStyles()
    let result_img_paths = props.result_img_paths
    let result_img_gallery_obj = convert_imgs_to_gallery_obj(result_img_paths)


    return (
        <Box className={classes.root}>
            <Typography variant="h5" className={classes.result_title}>Result</Typography>
            {/*<Box className={classes.result_gallery_wrapper}>*/}
            {/*    <ResponsiveGallery images={result_img_gallery_obj} useLightBox={true} />*/}
            {/*</Box>*/}
            <Grid container spacing="1" style={{width:"auto"}}>
                {result_img_paths.map((image_video_url)=> {
                    let content;
                    if (image_video_url.img.split(".").slice(-1)[0] === "png") {
                        content = (
                            <div className={classes.entry_box}>
                                <ModalImage small={image_video_url.img} large={image_video_url.img}/>
                                <Typography variant="subtitle1">{image_video_url.timestamp}</Typography>
                            </div>
                            )
                    } else {
                        content = (
                            <div className={classes.entry_box}>
                                <Player src={image_video_url.img}/>
                                <Typography variant="subtitle1">{image_video_url.timestamp}</Typography>
                            </div>
                        )
                    }
                    return (
                        <Grid item xs={4}>
                            {content}
                        </Grid>
                    )
                })}
            </Grid>
        </Box>
    )
}

export default ResultPanel;